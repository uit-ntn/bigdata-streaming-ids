from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

TEST_DATA_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression" / "logistic_regression.joblib"
DECISION_TREE_MODEL_PATH = MODEL_DIR / "decision_tree" / "decision_tree.joblib"
DEEP_MLP_MODEL_PATH = MODEL_DIR / "deep_mlp" / "deep_mlp_final.keras"

DROP_COLS = [
    "id",
    "attack_cat",
    "label",
]


# ============================================================
# 2. FLASK APP CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


# ============================================================
# 3. GLOBAL CACHE
# ============================================================

CACHE = {
    "test_df": None,
    "preprocessor": None,
    "models": {},
}


# ============================================================
# 4. LOAD DATA / MODELS
# ============================================================

def load_test_data():
    if CACHE["test_df"] is not None:
        return CACHE["test_df"]

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing testing data: {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    CACHE["test_df"] = df

    return df


def load_preprocessor():
    if CACHE["preprocessor"] is not None:
        return CACHE["preprocessor"]

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing preprocessor: {PREPROCESSOR_PATH}. "
            "Please run: python src/preprocessing.py"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    CACHE["preprocessor"] = preprocessor

    return preprocessor


def get_available_models():
    available_models = {}

    if LOGISTIC_MODEL_PATH.exists():
        available_models["logistic_regression"] = {
            "display_name": "Logistic Regression",
            "type": "sklearn",
            "path": LOGISTIC_MODEL_PATH,
        }

    if DECISION_TREE_MODEL_PATH.exists():
        available_models["decision_tree"] = {
            "display_name": "Decision Tree",
            "type": "sklearn",
            "path": DECISION_TREE_MODEL_PATH,
        }

    if DEEP_MLP_MODEL_PATH.exists():
        available_models["deep_mlp"] = {
            "display_name": "Deep MLP",
            "type": "keras",
            "path": DEEP_MLP_MODEL_PATH,
        }

    return available_models


def load_model(model_key):
    available_models = get_available_models()

    if model_key not in available_models:
        raise ValueError(f"Model is not available: {model_key}")

    if model_key in CACHE["models"]:
        return CACHE["models"][model_key], available_models[model_key]

    model_info = available_models[model_key]

    if model_info["type"] == "sklearn":
        model = joblib.load(model_info["path"])

    elif model_info["type"] == "keras":
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TensorFlow is not installed. Install it with: pip install tensorflow"
            )

        model = tf.keras.models.load_model(model_info["path"])

    else:
        raise ValueError(f"Unsupported model type: {model_info['type']}")

    CACHE["models"][model_key] = model

    return model, model_info


# ============================================================
# 5. SCHEMA / FORM HELPERS
# ============================================================

def get_feature_columns(test_df):
    return [col for col in test_df.columns if col not in DROP_COLS]


def get_column_groups(test_df):
    feature_cols = get_feature_columns(test_df)

    categorical_cols = (
        test_df[feature_cols]
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numerical_cols = (
        test_df[feature_cols]
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    other_cols = [
        col for col in feature_cols
        if col not in categorical_cols and col not in numerical_cols
    ]

    return categorical_cols, numerical_cols, other_cols


def get_categorical_options(test_df, categorical_cols, max_options=80):
    options = {}

    for col in categorical_cols:
        values = (
            test_df[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(max_options)
            .index
            .tolist()
        )

        options[col] = values

    return options


def get_template_row(test_df, row_index):
    row_index = int(row_index)

    if row_index < 0:
        row_index = 0

    if row_index >= len(test_df):
        row_index = len(test_df) - 1

    return test_df.iloc[row_index]


def build_manual_input_df(test_df, form_data):
    feature_cols = get_feature_columns(test_df)
    categorical_cols, numerical_cols, other_cols = get_column_groups(test_df)

    input_data = {}

    for col in categorical_cols:
        input_data[col] = form_data.get(col, "")

    for col in numerical_cols:
        raw_value = form_data.get(col, "")

        if raw_value is None or raw_value == "":
            raw_value = 0

        try:
            if pd.api.types.is_integer_dtype(test_df[col]):
                input_data[col] = int(float(raw_value))
            else:
                input_data[col] = float(raw_value)
        except ValueError:
            input_data[col] = 0

    for col in other_cols:
        input_data[col] = form_data.get(col, "")

    manual_df = pd.DataFrame([input_data], columns=feature_cols)

    return manual_df


def build_sample_df(test_df, input_mode, sample_size, random_state):
    sample_size = int(sample_size)
    random_state = int(random_state)

    if input_mode == "first_n":
        return test_df.head(sample_size).copy()

    if input_mode == "random":
        return test_df.sample(
            n=min(sample_size, len(test_df)),
            random_state=random_state,
        ).copy()

    if input_mode == "attack_only":
        if "label" not in test_df.columns:
            raise ValueError("Column 'label' not found in testing data.")

        attack_df = test_df[test_df["label"] == 1]

        return attack_df.sample(
            n=min(sample_size, len(attack_df)),
            random_state=random_state,
        ).copy()

    if input_mode == "normal_only":
        if "label" not in test_df.columns:
            raise ValueError("Column 'label' not found in testing data.")

        normal_df = test_df[test_df["label"] == 0]

        return normal_df.sample(
            n=min(sample_size, len(normal_df)),
            random_state=random_state,
        ).copy()

    raise ValueError(f"Unsupported input mode: {input_mode}")


# ============================================================
# 6. PREDICTION HELPERS
# ============================================================

def prepare_features(raw_df):
    drop_cols = [col for col in DROP_COLS if col in raw_df.columns]
    X = raw_df.drop(columns=drop_cols)

    return X, drop_cols


def predict_with_model(model, model_type, X_processed, threshold):
    start_time = time.time()

    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            attack_probability = model.predict_proba(X_processed)[:, 1]
        else:
            decision_score = model.decision_function(X_processed)
            attack_probability = 1 / (1 + np.exp(-decision_score))

        y_pred = (attack_probability >= threshold).astype(int)

    elif model_type == "keras":
        X_dense = X_processed.astype(np.float32).toarray()
        attack_probability = model.predict(X_dense, verbose=0).ravel()
        y_pred = (attack_probability >= threshold).astype(int)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    prediction_time = time.time() - start_time

    return y_pred, attack_probability, prediction_time


def build_result_df(input_df, y_pred, attack_probability):
    result_df = input_df.copy()

    result_df["predicted_label"] = y_pred
    result_df["predicted_class"] = np.where(y_pred == 1, "Attack", "Normal")
    result_df["attack_probability"] = attack_probability

    if "label" in result_df.columns:
        result_df["actual_class"] = np.where(result_df["label"] == 1, "Attack", "Normal")
        result_df["is_correct"] = result_df["label"] == result_df["predicted_label"]

    return result_df


def summarize_result(result_df, preprocessing_time, prediction_time, model_display_name, threshold, input_mode):
    total_records = len(result_df)
    pred_attack = int((result_df["predicted_label"] == 1).sum())
    pred_normal = int((result_df["predicted_label"] == 0).sum())
    attack_ratio = pred_attack / total_records * 100 if total_records > 0 else 0

    summary = {
        "model": model_display_name,
        "threshold": threshold,
        "input_mode": input_mode,
        "total_records": total_records,
        "predicted_normal": pred_normal,
        "predicted_attack": pred_attack,
        "attack_ratio": attack_ratio,
        "preprocessing_time": preprocessing_time,
        "prediction_time": prediction_time,
        "prediction_time_per_record_ms": prediction_time / total_records * 1000 if total_records > 0 else 0,
    }

    if "is_correct" in result_df.columns:
        correct = int(result_df["is_correct"].sum())
        accuracy = correct / total_records * 100 if total_records > 0 else 0

        summary["correct_predictions"] = correct
        summary["sample_accuracy"] = accuracy

    return summary


def make_display_table(result_df, max_rows=30):
    priority_cols = [
        "predicted_class",
        "predicted_label",
        "attack_probability",
    ]

    if "actual_class" in result_df.columns:
        priority_cols = [
            "actual_class",
            "predicted_class",
            "is_correct",
            "predicted_label",
            "attack_probability",
        ]

    remaining_cols = [col for col in result_df.columns if col not in priority_cols]

    display_df = result_df[priority_cols + remaining_cols].head(max_rows).copy()

    if "attack_probability" in display_df.columns:
        display_df["attack_probability"] = display_df["attack_probability"].map(lambda x: f"{x:.6f}")

    return display_df


def get_prediction_distribution(result_df):
    counts = result_df["predicted_class"].value_counts().to_dict()

    normal = int(counts.get("Normal", 0))
    attack = int(counts.get("Attack", 0))
    total = normal + attack

    return {
        "normal": normal,
        "attack": attack,
        "normal_percent": normal / total * 100 if total > 0 else 0,
        "attack_percent": attack / total * 100 if total > 0 else 0,
    }


def get_probability_bins(result_df):
    bins = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]

    rows = []

    for low, high in bins:
        if high == 1.0:
            count = int(((result_df["attack_probability"] >= low) & (result_df["attack_probability"] <= high)).sum())
        else:
            count = int(((result_df["attack_probability"] >= low) & (result_df["attack_probability"] < high)).sum())

        rows.append({
            "label": f"{low:.1f}-{high:.1f}",
            "count": count,
        })

    max_count = max([row["count"] for row in rows], default=1)

    for row in rows:
        row["percent_width"] = row["count"] / max_count * 100 if max_count > 0 else 0

    return rows


def get_actual_vs_predicted(result_df):
    if "actual_class" not in result_df.columns:
        return None

    matrix = pd.crosstab(
        result_df["actual_class"],
        result_df["predicted_class"],
    )

    for row_name in ["Normal", "Attack"]:
        if row_name not in matrix.index:
            matrix.loc[row_name] = 0

    for col_name in ["Normal", "Attack"]:
        if col_name not in matrix.columns:
            matrix[col_name] = 0

    matrix = matrix.loc[["Normal", "Attack"], ["Normal", "Attack"]]

    return {
        "normal_normal": int(matrix.loc["Normal", "Normal"]),
        "normal_attack": int(matrix.loc["Normal", "Attack"]),
        "attack_normal": int(matrix.loc["Attack", "Normal"]),
        "attack_attack": int(matrix.loc["Attack", "Attack"]),
    }


# ============================================================
# 7. ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "available_models": get_available_models(),
        "selected_model": "logistic_regression",
        "input_mode": "manual",
        "threshold": 0.5,
        "sample_size": 100,
        "random_state": 42,
        "template_row_index": 0,
        "test_shape": None,
        "test_label_summary": None,
        "categorical_cols": [],
        "numerical_cols": [],
        "categorical_options": {},
        "manual_defaults": {},
        "summary": None,
        "display_table": None,
        "prediction_distribution": None,
        "probability_bins": None,
        "actual_vs_predicted": None,
    }

    try:
        test_df = load_test_data()

        context["test_shape"] = test_df.shape

        if "label" in test_df.columns:
            context["test_label_summary"] = {
                "normal": int((test_df["label"] == 0).sum()),
                "attack": int((test_df["label"] == 1).sum()),
            }

        categorical_cols, numerical_cols, other_cols = get_column_groups(test_df)
        context["categorical_cols"] = categorical_cols
        context["numerical_cols"] = numerical_cols
        context["categorical_options"] = get_categorical_options(test_df, categorical_cols)

        if request.method == "POST":
            selected_model = request.form.get("selected_model", "logistic_regression")
            input_mode = request.form.get("input_mode", "manual")
            threshold = float(request.form.get("threshold", 0.5))
            sample_size = int(request.form.get("sample_size", 100))
            random_state = int(request.form.get("random_state", 42))
            template_row_index = int(request.form.get("template_row_index", 0))

            context["selected_model"] = selected_model
            context["input_mode"] = input_mode
            context["threshold"] = threshold
            context["sample_size"] = sample_size
            context["random_state"] = random_state
            context["template_row_index"] = template_row_index

            if input_mode == "manual":
                input_df = build_manual_input_df(test_df, request.form)
            else:
                input_df = build_sample_df(test_df, input_mode, sample_size, random_state)

            preprocessor = load_preprocessor()
            X_raw, dropped_cols = prepare_features(input_df)

            start_preprocess = time.time()
            X_processed = preprocessor.transform(X_raw)
            preprocessing_time = time.time() - start_preprocess

            model, model_info = load_model(selected_model)

            y_pred, attack_probability, prediction_time = predict_with_model(
                model=model,
                model_type=model_info["type"],
                X_processed=X_processed,
                threshold=threshold,
            )

            result_df = build_result_df(input_df, y_pred, attack_probability)

            context["summary"] = summarize_result(
                result_df=result_df,
                preprocessing_time=preprocessing_time,
                prediction_time=prediction_time,
                model_display_name=model_info["display_name"],
                threshold=threshold,
                input_mode=input_mode,
            )

            context["display_table"] = make_display_table(result_df).to_dict(orient="records")
            context["display_columns"] = make_display_table(result_df).columns.tolist()
            context["prediction_distribution"] = get_prediction_distribution(result_df)
            context["probability_bins"] = get_probability_bins(result_df)
            context["actual_vs_predicted"] = get_actual_vs_predicted(result_df)

        else:
            template_row = get_template_row(test_df, 0)
            feature_cols = get_feature_columns(test_df)

            context["manual_defaults"] = {
                col: template_row[col]
                for col in feature_cols
            }

        if request.method == "POST" and context["input_mode"] == "manual":
            context["manual_defaults"] = {
                col: request.form.get(col, "")
                for col in get_feature_columns(test_df)
            }
        elif request.method == "POST":
            template_row = get_template_row(test_df, context["template_row_index"])
            context["manual_defaults"] = {
                col: template_row[col]
                for col in get_feature_columns(test_df)
            }

    except Exception as e:
        context["error"] = str(e)
        context["traceback"] = traceback.format_exc()

    return render_template("index.html", **context)


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )