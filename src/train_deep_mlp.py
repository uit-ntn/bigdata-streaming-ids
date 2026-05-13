from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics as keras_metrics

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_NAME = "deep_mlp"

MODEL_DIR = BASE_DIR / "models" / MODEL_NAME

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "models" / MODEL_NAME
RESULT_DIR = REPORT_DIR / "results" / "models" / MODEL_NAME

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "model_name": MODEL_NAME,
    "algorithm": "Deep Neural Network - Multi-Layer Perceptron",
    "task": "Binary Classification",
    "target": "label",
    "label_mapping": {
        "0": "Normal",
        "1": "Attack",
    },
    "architecture": {
        "input": "processed_feature_vector",
        "hidden_layers": [
            {
                "units": 256,
                "activation": "relu",
                "batch_normalization": True,
                "dropout": 0.30,
            },
            {
                "units": 128,
                "activation": "relu",
                "batch_normalization": True,
                "dropout": 0.30,
            },
            {
                "units": 64,
                "activation": "relu",
                "batch_normalization": False,
                "dropout": 0.20,
            },
        ],
        "output": {
            "units": 1,
            "activation": "sigmoid",
        },
    },
    "training_parameters": {
        "epochs": 40,
        "batch_size": 512,
        "validation_split": 0.20,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "binary_crossentropy",
        "early_stopping_patience": 6,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.5,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "note": (
        "MLP is used because UNSW-NB15 after preprocessing is tabular feature-vector data. "
        "The model learns non-linear relationships between numerical and one-hot encoded categorical features."
    ),
}


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(filename: str):
    plt.tight_layout()
    save_path = FIGURE_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 4. LOAD PROCESSED DATA
# ============================================================

def load_processed_data():
    print_section("1. LOAD PROCESSED DATA")

    X_train_path = PROCESSED_DIR / "X_train_processed.npz"
    X_test_path = PROCESSED_DIR / "X_test_processed.npz"
    y_train_path = PROCESSED_DIR / "y_train.npy"
    y_test_path = PROCESSED_DIR / "y_test.npy"
    feature_names_path = PROCESSED_DIR / "feature_names.json"
    column_info_path = PROCESSED_DIR / "column_info.json"

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}. Please run src/preprocessing.py first."
            )

    X_train_sparse = sparse.load_npz(X_train_path)
    X_test_sparse = sparse.load_npz(X_test_path)

    y_train = np.load(y_train_path).astype(np.float32)
    y_test = np.load(y_test_path).astype(np.float32)

    feature_names = load_json_if_exists(feature_names_path)
    column_info = load_json_if_exists(column_info_path)

    print(f"X_train sparse shape: {X_train_sparse.shape}")
    print(f"X_test sparse shape : {X_test_sparse.shape}")
    print(f"y_train shape       : {y_train.shape}")
    print(f"y_test shape        : {y_test.shape}")

    if feature_names is not None:
        print(f"Number of feature names: {len(feature_names)}")
    else:
        print("[WARNING] feature_names.json not found.")

    return X_train_sparse, X_test_sparse, y_train, y_test, feature_names, column_info


# ============================================================
# 5. CONVERT SPARSE TO DENSE
# ============================================================

def convert_to_dense_float32(X_train_sparse, X_test_sparse):
    print_section("2. CONVERT SPARSE MATRIX TO DENSE FLOAT32")

    print("Deep Learning models require dense tensors.")
    print("Converting sparse matrices to dense float32 arrays...")

    start_time = time.time()

    try:
        X_train = X_train_sparse.astype(np.float32).toarray()
        X_test = X_test_sparse.astype(np.float32).toarray()
    except MemoryError:
        raise MemoryError(
            "Not enough RAM to convert sparse matrix to dense. "
            "Reduce dataset size or use a TensorFlow sparse input pipeline."
        )

    convert_time = time.time() - start_time

    print(f"X_train dense shape: {X_train.shape}")
    print(f"X_test dense shape : {X_test.shape}")
    print(f"Conversion time    : {convert_time:.4f} seconds")
    print(f"X_train memory     : {X_train.nbytes / (1024 ** 2):.2f} MB")
    print(f"X_test memory      : {X_test.nbytes / (1024 ** 2):.2f} MB")

    dense_summary = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "X_train_memory_mb": X_train.nbytes / (1024 ** 2),
        "X_test_memory_mb": X_test.nbytes / (1024 ** 2),
        "conversion_time_seconds": convert_time,
    }

    save_json(dense_summary, RESULT_DIR / "dense_conversion_summary.json")

    return X_train, X_test


# ============================================================
# 6. DATA SUMMARY
# ============================================================

def summarize_training_data(X_train, X_test, y_train, y_test, feature_names, column_info):
    print_section("3. DATA SUMMARY")

    train_label_counts = pd.Series(y_train.astype(int)).value_counts().sort_index()
    test_label_counts = pd.Series(y_test.astype(int)).value_counts().sort_index()

    print("Train label distribution:")
    print(train_label_counts)

    print("\nTest label distribution:")
    print(test_label_counts)

    summary = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "train_label_distribution": train_label_counts.to_dict(),
        "test_label_distribution": test_label_counts.to_dict(),
        "num_features": X_train.shape[1],
        "has_feature_names": feature_names is not None,
        "column_info": column_info,
    }

    save_json(summary, RESULT_DIR / "data_summary.json")

    label_df = pd.DataFrame({
        "dataset": ["train", "train", "test", "test"],
        "label": [0, 1, 0, 1],
        "label_name": ["Normal", "Attack", "Normal", "Attack"],
        "count": [
            int((y_train == 0).sum()),
            int((y_train == 1).sum()),
            int((y_test == 0).sum()),
            int((y_test == 1).sum()),
        ],
    })

    label_df["ratio_percent"] = label_df.apply(
        lambda row: row["count"] / len(y_train) * 100
        if row["dataset"] == "train"
        else row["count"] / len(y_test) * 100,
        axis=1,
    )

    label_df.to_csv(RESULT_DIR / "label_distribution.csv", index=False)

    return summary


# ============================================================
# 7. CLASS WEIGHT
# ============================================================

def compute_training_class_weight(y_train):
    print_section("4. COMPUTE CLASS WEIGHT")

    classes = np.unique(y_train.astype(int))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int),
    )

    class_weight_dict = {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }

    print("Class weights:")
    print(class_weight_dict)

    save_json(class_weight_dict, RESULT_DIR / "class_weight.json")

    return class_weight_dict


# ============================================================
# 8. BUILD MODEL
# ============================================================

def build_model(input_dim: int):
    print_section("5. BUILD DEEP MLP MODEL")

    set_random_seed(MODEL_CONFIG["training_parameters"]["random_state"])

    learning_rate = MODEL_CONFIG["training_parameters"]["learning_rate"]

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_features"),

            layers.Dense(256, activation="relu", name="dense_256"),
            layers.BatchNormalization(name="batch_norm_256"),
            layers.Dropout(0.30, name="dropout_256"),

            layers.Dense(128, activation="relu", name="dense_128"),
            layers.BatchNormalization(name="batch_norm_128"),
            layers.Dropout(0.30, name="dropout_128"),

            layers.Dense(64, activation="relu", name="dense_64"),
            layers.Dropout(0.20, name="dropout_64"),

            layers.Dense(1, activation="sigmoid", name="output_attack_probability"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras_metrics.Precision(name="precision"),
            keras_metrics.Recall(name="recall"),
            keras_metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    save_json(MODEL_CONFIG, RESULT_DIR / "model_config.json")

    architecture_summary = []

    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "trainable": layer.trainable,
            "num_params": layer.count_params(),
        }
        architecture_summary.append(layer_info)

    architecture_df = pd.DataFrame(architecture_summary)
    architecture_df.to_csv(RESULT_DIR / "model_architecture_summary.csv", index=False)

    save_json(architecture_summary, RESULT_DIR / "model_architecture_summary.json")

    return model


# ============================================================
# 9. CALLBACKS
# ============================================================

def build_callbacks():
    print_section("6. BUILD CALLBACKS")

    best_model_path = MODEL_DIR / "deep_mlp_best.keras"

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=MODEL_CONFIG["training_parameters"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=MODEL_CONFIG["training_parameters"]["reduce_lr_factor"],
            patience=MODEL_CONFIG["training_parameters"]["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    callback_summary = {
        "early_stopping": {
            "monitor": "val_loss",
            "patience": MODEL_CONFIG["training_parameters"]["early_stopping_patience"],
            "restore_best_weights": True,
        },
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": MODEL_CONFIG["training_parameters"]["reduce_lr_factor"],
            "patience": MODEL_CONFIG["training_parameters"]["reduce_lr_patience"],
            "min_lr": 1e-6,
        },
        "model_checkpoint": {
            "monitor": "val_loss",
            "save_best_only": True,
            "path": str(best_model_path),
        },
    }

    save_json(callback_summary, RESULT_DIR / "callback_summary.json")

    print("Callbacks configured:")
    print(json.dumps(callback_summary, indent=4, ensure_ascii=False))

    return callback_list


# ============================================================
# 10. TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train, class_weight_dict):
    print_section("7. TRAIN MODEL")

    epochs = MODEL_CONFIG["training_parameters"]["epochs"]
    batch_size = MODEL_CONFIG["training_parameters"]["batch_size"]
    validation_split = MODEL_CONFIG["training_parameters"]["validation_split"]

    callback_list = build_callbacks()

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight_dict,
        callbacks=callback_list,
        verbose=1,
    )

    train_time = time.time() - start_time

    print(f"[OK] Training completed in {train_time:.4f} seconds.")

    final_model_path = MODEL_DIR / "deep_mlp_final.keras"
    model.save(final_model_path)

    print(f"[OK] Saved final model: {final_model_path}")

    history_df = pd.DataFrame(history.history)
    history_df.index.name = "epoch"
    history_df.to_csv(RESULT_DIR / "training_history.csv")

    history_path = MODEL_DIR / "deep_mlp_history.joblib"
    joblib.dump(history.history, history_path)

    training_info = {
        "train_time_seconds": train_time,
        "epochs_configured": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "validation_split": validation_split,
        "final_model_path": str(final_model_path),
        "history_path": str(history_path),
    }

    save_json(training_info, RESULT_DIR / "training_info.json")

    return model, history, train_time


# ============================================================
# 11. PREDICT
# ============================================================

def predict_model(model, X_test):
    print_section("8. PREDICT ON TEST SET")

    start_time = time.time()
    y_proba = model.predict(X_test, batch_size=1024).ravel()
    prediction_time = time.time() - start_time

    y_pred = (y_proba >= 0.5).astype(int)

    print(f"[OK] Prediction completed in {prediction_time:.4f} seconds.")

    prediction_info = {
        "prediction_time_seconds": prediction_time,
        "num_test_samples": X_test.shape[0],
        "prediction_time_per_sample_ms": prediction_time / X_test.shape[0] * 1000,
        "threshold": 0.5,
    }

    save_json(prediction_info, RESULT_DIR / "prediction_time_info.json")

    prediction_df = pd.DataFrame({
        "y_pred": y_pred,
        "attack_probability": y_proba,
    })

    prediction_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    return y_pred, y_proba, prediction_time


# ============================================================
# 12. EVALUATION
# ============================================================

def evaluate_model(y_test, y_pred, y_proba, train_time, prediction_time):
    print_section("9. EVALUATE MODEL")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)

    precision_attack = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall_attack = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1_attack = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "model": MODEL_NAME,
        "accuracy": accuracy,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_time_seconds": train_time,
        "prediction_time_seconds": prediction_time,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULT_DIR / "metrics_summary.csv", index=False)
    save_json(metrics, RESULT_DIR / "metrics_summary.json")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
    )

    report_path = RESULT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("Classification report:")
    print(report)

    print("Metrics summary:")
    print(metrics_df.T)

    return metrics, cm


# ============================================================
# 13. VISUALIZATIONS - TRAINING HISTORY
# ============================================================

def plot_training_loss(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train Loss")

    if "val_loss" in history_df.columns:
        plt.plot(history_df["val_loss"], label="Validation Loss")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("01_training_loss_curve.png")


def plot_training_accuracy(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Train Accuracy")

    if "val_accuracy" in history_df.columns:
        plt.plot(history_df["val_accuracy"], label="Validation Accuracy")

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("02_training_accuracy_curve.png")


def plot_training_precision_recall(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))

    if "precision" in history_df.columns:
        plt.plot(history_df["precision"], label="Train Precision")

    if "recall" in history_df.columns:
        plt.plot(history_df["recall"], label="Train Recall")

    if "val_precision" in history_df.columns:
        plt.plot(history_df["val_precision"], label="Validation Precision")

    if "val_recall" in history_df.columns:
        plt.plot(history_df["val_recall"], label="Validation Recall")

    plt.title("Training Precision and Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("03_training_precision_recall_curve.png")


def plot_training_auc(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))

    if "auc" in history_df.columns:
        plt.plot(history_df["auc"], label="Train AUC")

    if "val_auc" in history_df.columns:
        plt.plot(history_df["val_auc"], label="Validation AUC")

    plt.title("Training and Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("04_training_auc_curve.png")


# ============================================================
# 14. VISUALIZATIONS - EVALUATION
# ============================================================

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Deep MLP")
    plt.colorbar()

    labels = ["Normal", "Attack"]
    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    save_figure("05_confusion_matrix.png")


def plot_normalized_confusion_matrix(cm):
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("Normalized Confusion Matrix - Deep MLP")
    plt.colorbar()

    labels = ["Normal", "Attack"]
    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    save_figure("06_normalized_confusion_matrix.png")


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    roc_df = pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": thresholds,
    })

    roc_df.to_csv(RESULT_DIR / "roc_curve_points.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    plt.title("ROC Curve - Deep MLP")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()

    save_figure("07_roc_curve.png")


def plot_precision_recall_curve(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    pr_df = pd.DataFrame({
        "precision": precision[:-1],
        "recall": recall[:-1],
        "threshold": thresholds,
    })

    pr_df.to_csv(RESULT_DIR / "precision_recall_curve_points.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")

    plt.title("Precision-Recall Curve - Deep MLP")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    save_figure("08_precision_recall_curve.png")


def plot_probability_distribution(y_test, y_proba):
    normal_proba = y_proba[y_test == 0]
    attack_proba = y_proba[y_test == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_proba, bins=50, alpha=0.6, label="Normal")
    plt.hist(attack_proba, bins=50, alpha=0.6, label="Attack")

    plt.title("Predicted Attack Probability Distribution")
    plt.xlabel("Predicted Probability of Attack")
    plt.ylabel("Frequency")
    plt.legend()

    save_figure("09_probability_distribution.png")


def threshold_analysis(y_test, y_proba):
    thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision_attack": precision_score(
                y_test,
                y_pred_threshold,
                pos_label=1,
                zero_division=0,
            ),
            "recall_attack": recall_score(
                y_test,
                y_pred_threshold,
                pos_label=1,
                zero_division=0,
            ),
            "f1_attack": f1_score(
                y_test,
                y_pred_threshold,
                pos_label=1,
                zero_division=0,
            ),
            "accuracy": accuracy_score(y_test, y_pred_threshold),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(RESULT_DIR / "threshold_analysis.csv", index=False)

    return threshold_df


def plot_threshold_metrics(threshold_df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision_attack"],
        marker="o",
        label="Precision Attack",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall_attack"],
        marker="o",
        label="Recall Attack",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1_attack"],
        marker="o",
        label="F1 Attack",
    )

    plt.title("Threshold Analysis - Deep MLP")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("10_threshold_metrics.png")


def plot_prediction_summary_donut(y_pred):
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    labels = []
    values = []

    for label in [0, 1]:
        labels.append("Normal" if label == 0 else "Attack")
        values.append(int(pred_counts.get(label, 0)))

    plt.figure(figsize=(7, 6))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.42},
    )

    plt.text(
        0,
        0,
        f"{len(y_pred):,}\npredictions",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.title("Prediction Distribution - Deep MLP")

    save_figure("11_prediction_distribution_donut.png")

    pred_df = pd.DataFrame({
        "predicted_label": [0, 1],
        "label_name": labels,
        "count": values,
        "ratio_percent": [value / len(y_pred) * 100 for value in values],
    })

    pred_df.to_csv(RESULT_DIR / "prediction_distribution.csv", index=False)


def plot_metrics_summary_table(metrics):
    selected_metrics = [
        ["Accuracy", metrics["accuracy"]],
        ["Precision Attack", metrics["precision_attack"]],
        ["Recall Attack", metrics["recall_attack"]],
        ["F1 Attack", metrics["f1_attack"]],
        ["ROC-AUC", metrics["roc_auc"]],
        ["PR-AUC", metrics["pr_auc"]],
        ["Train time (s)", metrics["train_time_seconds"]],
        ["Prediction time (s)", metrics["prediction_time_seconds"]],
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    table_data = [
        [name, f"{value:.4f}" if isinstance(value, float) else value]
        for name, value in selected_metrics
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    plt.title("Deep MLP Metrics Summary")

    save_path = FIGURE_DIR / "12_metrics_summary_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {save_path}")


# ============================================================
# 15. FEATURE IMPORTANCE APPROXIMATION
# ============================================================

def approximate_input_weight_importance(model, feature_names):
    print_section("10. APPROXIMATE FEATURE IMPORTANCE FROM INPUT WEIGHTS")

    if feature_names is None:
        print("[WARNING] Skip input weight importance because feature_names is missing.")
        return

    first_dense_layer = None

    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            first_dense_layer = layer
            break

    if first_dense_layer is None:
        print("[WARNING] No Dense layer found.")
        return

    weights = first_dense_layer.get_weights()[0]

    importance = np.mean(np.abs(weights), axis=1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "approx_importance": importance,
    })

    importance_df = importance_df.sort_values("approx_importance", ascending=False)
    importance_df.to_csv(RESULT_DIR / "approx_input_weight_feature_importance.csv", index=False)

    top_df = importance_df.head(25).sort_values("approx_importance", ascending=True)

    plt.figure(figsize=(10, 9))
    plt.barh(top_df["feature"], top_df["approx_importance"])

    plt.title("Approximate Feature Importance from First Dense Layer")
    plt.xlabel("Mean Absolute Input Weight")
    plt.ylabel("Feature")

    save_figure("13_approx_input_weight_feature_importance.png")

    top_lollipop_df = importance_df.head(20).sort_values("approx_importance", ascending=True)

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_lollipop_df))

    plt.hlines(
        y=y_pos,
        xmin=0,
        xmax=top_lollipop_df["approx_importance"],
    )

    plt.plot(top_lollipop_df["approx_importance"], y_pos, "o")
    plt.yticks(y_pos, top_lollipop_df["feature"])
    plt.title("Top 20 Approximate Feature Importance - Lollipop Chart")
    plt.xlabel("Mean Absolute Input Weight")
    plt.ylabel("Feature")

    save_figure("14_approx_feature_importance_lollipop.png")


# ============================================================
# 16. CREATE VISUALIZATIONS
# ============================================================

def create_visualizations(model, history, y_test, y_pred, y_proba, cm, metrics, feature_names):
    print_section("11. CREATE VISUALIZATIONS")

    plot_training_loss(history)
    plot_training_accuracy(history)
    plot_training_precision_recall(history)
    plot_training_auc(history)

    plot_confusion_matrix(cm)
    plot_normalized_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_probability_distribution(y_test, y_proba)

    threshold_df = threshold_analysis(y_test, y_proba)
    plot_threshold_metrics(threshold_df)

    plot_prediction_summary_donut(y_pred)
    plot_metrics_summary_table(metrics)

    approximate_input_weight_importance(model, feature_names)


# ============================================================
# 17. GENERATE REPORT NOTES
# ============================================================

def generate_report_notes(metrics):
    print_section("12. GENERATE REPORT NOTES")

    lines = [
        "# Deep MLP Report Notes",
        "",
        "## 1. Mục tiêu",
        "",
        "Mô hình Deep MLP được sử dụng làm mô hình Deep Learning cho bài toán phát hiện xâm nhập mạng.",
        "Dữ liệu UNSW-NB15 sau tiền xử lý được biểu diễn dưới dạng vector đặc trưng, vì vậy kiến trúc MLP phù hợp hơn CNN/LSTM trong phạm vi hiện tại.",
        "",
        "## 2. Cấu hình mô hình",
        "",
        f"Model: {MODEL_CONFIG['algorithm']}",
        "Kiến trúc:",
        "- Dense 256 + ReLU + BatchNormalization + Dropout 0.30",
        "- Dense 128 + ReLU + BatchNormalization + Dropout 0.30",
        "- Dense 64 + ReLU + Dropout 0.20",
        "- Dense 1 + Sigmoid",
        "",
        f"Optimizer: {MODEL_CONFIG['training_parameters']['optimizer']}",
        f"Learning rate: {MODEL_CONFIG['training_parameters']['learning_rate']}",
        f"Loss: {MODEL_CONFIG['training_parameters']['loss']}",
        f"Epochs: {MODEL_CONFIG['training_parameters']['epochs']}",
        f"Batch size: {MODEL_CONFIG['training_parameters']['batch_size']}",
        f"Validation split: {MODEL_CONFIG['training_parameters']['validation_split']}",
        f"Class weight: {MODEL_CONFIG['training_parameters']['class_weight']}",
        "",
        "Class weight được sử dụng để giảm ảnh hưởng của mất cân bằng dữ liệu.",
        "EarlyStopping và ReduceLROnPlateau được sử dụng để hạn chế overfitting và điều chỉnh learning rate khi validation loss không cải thiện.",
        "",
        "## 3. Kết quả đánh giá",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision Attack: {metrics['precision_attack']:.4f}",
        f"Recall Attack: {metrics['recall_attack']:.4f}",
        f"F1-score Attack: {metrics['f1_attack']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"PR-AUC: {metrics['pr_auc']:.4f}",
        "",
        "Trong bài toán IDS, Recall của lớp Attack là chỉ số rất quan trọng vì bỏ sót tấn công có thể gây rủi ro bảo mật.",
        "",
        "## 4. Ma trận nhầm lẫn",
        "",
        f"True Negative: {metrics['tn']}",
        f"False Positive: {metrics['fp']}",
        f"False Negative: {metrics['fn']}",
        f"True Positive: {metrics['tp']}",
        "",
        "False Negative là các mẫu tấn công bị dự đoán nhầm thành bình thường, cần được chú ý khi đánh giá hệ thống IDS.",
        "",
        "## 5. Biểu đồ đã xuất",
        "",
        "Các biểu đồ được lưu tại:",
        "",
        "reports/figures/models/deep_mlp/",
        "",
        "Bao gồm:",
        "",
        "- Training loss curve",
        "- Training accuracy curve",
        "- Training precision/recall curve",
        "- Training AUC curve",
        "- Confusion matrix",
        "- Normalized confusion matrix",
        "- ROC curve",
        "- Precision-Recall curve",
        "- Probability distribution",
        "- Threshold analysis",
        "- Prediction distribution donut chart",
        "- Metrics summary table",
        "- Approximate feature importance from input weights",
        "",
        "## 6. Nhận xét",
        "",
        "Deep MLP có khả năng học quan hệ phi tuyến giữa các đặc trưng tốt hơn Logistic Regression.",
        "So với Decision Tree đơn lẻ, MLP có thể biểu diễn quan hệ phức tạp hơn nhưng khó giải thích trực tiếp hơn.",
        "Kết quả của Deep MLP được dùng để so sánh với hai baseline Logistic Regression và Decision Tree.",
    ]

    content = "\n".join(lines)

    report_path = RESULT_DIR / "deep_mlp_report_notes.md"
    report_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved report notes: {report_path}")


# ============================================================
# 18. MAIN
# ============================================================

def main():
    set_random_seed(MODEL_CONFIG["training_parameters"]["random_state"])

    X_train_sparse, X_test_sparse, y_train, y_test, feature_names, column_info = load_processed_data()

    X_train, X_test = convert_to_dense_float32(
        X_train_sparse=X_train_sparse,
        X_test_sparse=X_test_sparse,
    )

    summarize_training_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        column_info=column_info,
    )

    class_weight_dict = compute_training_class_weight(y_train)

    model = build_model(input_dim=X_train.shape[1])

    model, history, train_time = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        class_weight_dict=class_weight_dict,
    )

    y_pred, y_proba, prediction_time = predict_model(
        model=model,
        X_test=X_test,
    )

    metrics, cm = evaluate_model(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        train_time=train_time,
        prediction_time=prediction_time,
    )

    create_visualizations(
        model=model,
        history=history,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        cm=cm,
        metrics=metrics,
        feature_names=feature_names,
    )

    generate_report_notes(metrics)

    print("\nDeep MLP training completed successfully.")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()