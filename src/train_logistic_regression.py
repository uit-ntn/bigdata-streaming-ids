from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.linear_model import LogisticRegression
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

MODEL_NAME = "logistic_regression"

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
    "algorithm": "Logistic Regression",
    "task": "Binary Classification",
    "target": "label",
    "label_mapping": {
        "0": "Normal",
        "1": "Attack"
    },
    "parameters": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42
    },
    "note": (
        "Logistic Regression is used as a linear baseline model. "
        "class_weight='balanced' is used to reduce the effect of class imbalance."
    )
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


def save_figure(filename: str):
    plt.tight_layout()
    save_path = FIGURE_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


def load_json_if_exists(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    X_train = sparse.load_npz(X_train_path)
    X_test = sparse.load_npz(X_test_path)

    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    feature_names = load_json_if_exists(feature_names_path)
    column_info = load_json_if_exists(column_info_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    if feature_names is not None:
        print(f"Number of feature names: {len(feature_names)}")
    else:
        print("[WARNING] feature_names.json not found.")

    return X_train, X_test, y_train, y_test, feature_names, column_info


# ============================================================
# 5. DATA SUMMARY
# ============================================================

def summarize_training_data(X_train, X_test, y_train, y_test, feature_names, column_info):
    print_section("2. DATA SUMMARY")

    train_label_counts = pd.Series(y_train).value_counts().sort_index()
    test_label_counts = pd.Series(y_test).value_counts().sort_index()

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

    print("Train label distribution:")
    print(train_label_counts)

    print("\nTest label distribution:")
    print(test_label_counts)

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
# 6. BUILD MODEL
# ============================================================

def build_model():
    print_section("3. BUILD LOGISTIC REGRESSION MODEL")

    params = MODEL_CONFIG["parameters"]

    model = LogisticRegression(
        penalty=params["penalty"],
        C=params["C"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=params["random_state"],
    )

    save_json(MODEL_CONFIG, RESULT_DIR / "model_config.json")

    print("Model configuration:")
    print(json.dumps(MODEL_CONFIG, indent=4, ensure_ascii=False))

    return model


# ============================================================
# 7. TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train):
    print_section("4. TRAIN MODEL")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"[OK] Training completed in {train_time:.4f} seconds.")
    print(f"Number of iterations: {model.n_iter_}")

    model_path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    print(f"[OK] Saved model: {model_path}")

    training_info = {
        "train_time_seconds": train_time,
        "n_iter": model.n_iter_.tolist(),
        "classes": model.classes_.tolist(),
        "model_path": str(model_path),
    }

    save_json(training_info, RESULT_DIR / "training_info.json")

    return model, train_time


# ============================================================
# 8. PREDICT
# ============================================================

def predict_model(model, X_test):
    print_section("5. PREDICT ON TEST SET")

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    start_time = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]
    probability_time = time.time() - start_time

    print(f"[OK] Prediction completed in {prediction_time:.4f} seconds.")
    print(f"[OK] Probability prediction completed in {probability_time:.4f} seconds.")

    prediction_info = {
        "prediction_time_seconds": prediction_time,
        "probability_prediction_time_seconds": probability_time,
        "num_test_samples": X_test.shape[0],
        "prediction_time_per_sample_ms": prediction_time / X_test.shape[0] * 1000,
    }

    save_json(prediction_info, RESULT_DIR / "prediction_time_info.json")

    prediction_df = pd.DataFrame({
        "y_pred": y_pred,
        "attack_probability": y_proba,
    })

    prediction_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    return y_pred, y_proba, prediction_time


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(y_test, y_pred, y_proba, train_time, prediction_time):
    print_section("6. EVALUATE MODEL")

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
# 10. VISUALIZATIONS
# ============================================================

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Logistic Regression")
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

    save_figure("01_confusion_matrix.png")


def plot_normalized_confusion_matrix(cm):
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("Normalized Confusion Matrix - Logistic Regression")
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

    save_figure("02_normalized_confusion_matrix.png")


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

    plt.title("ROC Curve - Logistic Regression")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()

    save_figure("03_roc_curve.png")


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

    plt.title("Precision-Recall Curve - Logistic Regression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    save_figure("04_precision_recall_curve.png")


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

    save_figure("05_probability_distribution.png")


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

    plt.title("Threshold Analysis - Logistic Regression")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.3)

    save_figure("06_threshold_metrics.png")


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

    plt.title("Prediction Distribution - Logistic Regression")

    save_figure("07_prediction_distribution_donut.png")

    pred_df = pd.DataFrame({
        "predicted_label": [0, 1],
        "label_name": labels,
        "count": values,
        "ratio_percent": [value / len(y_pred) * 100 for value in values],
    })

    pred_df.to_csv(RESULT_DIR / "prediction_distribution.csv", index=False)


def plot_top_coefficients(model, feature_names):
    if feature_names is None:
        print("[WARNING] Skip coefficient plot because feature_names is missing.")
        return

    if not hasattr(model, "coef_"):
        print("[WARNING] Model has no coef_.")
        return

    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })

    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(RESULT_DIR / "logistic_regression_coefficients.csv", index=False)

    top_positive = coef_df.sort_values("coefficient", ascending=False).head(15)
    top_negative = coef_df.sort_values("coefficient", ascending=True).head(15)

    top_combined = pd.concat([top_negative, top_positive])
    top_combined = top_combined.sort_values("coefficient")

    plt.figure(figsize=(10, 8))
    plt.barh(top_combined["feature"], top_combined["coefficient"])

    plt.title("Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")

    save_figure("08_top_coefficients.png")


def plot_top_absolute_coefficients(model, feature_names):
    if feature_names is None:
        return

    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })

    coef_df = coef_df.sort_values("abs_coefficient", ascending=True).tail(20)

    plt.figure(figsize=(10, 8))
    plt.hlines(
        y=np.arange(len(coef_df)),
        xmin=0,
        xmax=coef_df["abs_coefficient"],
    )
    plt.plot(coef_df["abs_coefficient"], np.arange(len(coef_df)), "o")

    plt.yticks(np.arange(len(coef_df)), coef_df["feature"])
    plt.title("Top 20 Features by Absolute Coefficient")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")

    save_figure("09_top_absolute_coefficients_lollipop.png")


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

    plt.title("Logistic Regression Metrics Summary")

    save_path = FIGURE_DIR / "10_metrics_summary_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {save_path}")


def create_visualizations(model, y_test, y_pred, y_proba, cm, metrics, feature_names):
    print_section("7. CREATE VISUALIZATIONS")

    plot_confusion_matrix(cm)
    plot_normalized_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_probability_distribution(y_test, y_proba)

    threshold_df = threshold_analysis(y_test, y_proba)
    plot_threshold_metrics(threshold_df)

    plot_prediction_summary_donut(y_pred)
    plot_top_coefficients(model, feature_names)
    plot_top_absolute_coefficients(model, feature_names)
    plot_metrics_summary_table(metrics)


# ============================================================
# 11. GENERATE REPORT NOTES
# ============================================================

def generate_report_notes(metrics):
    print_section("8. GENERATE REPORT NOTES")

    lines = [
        "# Logistic Regression Report Notes",
        "",
        "## 1. Mục tiêu",
        "",
        "Mô hình Logistic Regression được sử dụng làm baseline tuyến tính cho bài toán phát hiện xâm nhập mạng.",
        "Mục tiêu là phân loại lưu lượng mạng thành hai lớp: Normal và Attack.",
        "",
        "## 2. Cấu hình mô hình",
        "",
        f"Model: {MODEL_CONFIG['algorithm']}",
        f"Penalty: {MODEL_CONFIG['parameters']['penalty']}",
        f"C: {MODEL_CONFIG['parameters']['C']}",
        f"Solver: {MODEL_CONFIG['parameters']['solver']}",
        f"Max iterations: {MODEL_CONFIG['parameters']['max_iter']}",
        f"Class weight: {MODEL_CONFIG['parameters']['class_weight']}",
        "",
        "Tham số `class_weight='balanced'` được sử dụng để giảm ảnh hưởng của mất cân bằng nhãn.",
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
        "Trong bài toán IDS, Recall của lớp Attack là chỉ số quan trọng vì bỏ sót tấn công có thể gây rủi ro bảo mật.",
        "",
        "## 4. Ma trận nhầm lẫn",
        "",
        f"True Negative: {metrics['tn']}",
        f"False Positive: {metrics['fp']}",
        f"False Negative: {metrics['fn']}",
        f"True Positive: {metrics['tp']}",
        "",
        "False Negative là các mẫu tấn công bị dự đoán nhầm thành bình thường, cần được quan tâm khi đánh giá hệ thống IDS.",
        "",
        "## 5. Biểu đồ đã xuất",
        "",
        "Các biểu đồ được lưu tại:",
        "",
        "reports/figures/logistic_regression/",
        "",
        "Bao gồm:",
        "",
        "- Confusion matrix",
        "- Normalized confusion matrix",
        "- ROC curve",
        "- Precision-Recall curve",
        "- Probability distribution",
        "- Threshold analysis",
        "- Prediction distribution donut chart",
        "- Top coefficients",
        "- Metrics summary table",
        "",
        "## 6. Nhận xét",
        "",
        "Logistic Regression là mô hình đơn giản, dễ giải thích và có tốc độ huấn luyện nhanh.",
        "Tuy nhiên, vì đây là mô hình tuyến tính nên khả năng học các quan hệ phi tuyến trong dữ liệu mạng có thể bị hạn chế.",
        "Kết quả từ mô hình này được dùng làm mốc so sánh với các mô hình mạnh hơn như Decision Tree, Random Forest, XGBoost hoặc LightGBM.",
    ]

    content = "\n".join(lines)

    report_path = RESULT_DIR / "logistic_regression_report_notes.md"
    report_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved report notes: {report_path}")


# ============================================================
# 12. MAIN
# ============================================================

def main():
    X_train, X_test, y_train, y_test, feature_names, column_info = load_processed_data()

    summarize_training_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        column_info=column_info,
    )

    model = build_model()

    model, train_time = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
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
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        cm=cm,
        metrics=metrics,
        feature_names=feature_names,
    )

    generate_report_notes(metrics)

    print("\nLogistic Regression training completed successfully.")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()