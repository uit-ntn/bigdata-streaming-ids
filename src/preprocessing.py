from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "results"
FIGURE_DIR = BASE_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

TARGET_COL = "label"

# id: mã định danh dòng dữ liệu, không có ý nghĩa dự đoán
# attack_cat: loại tấn công cụ thể, không dùng khi train binary label
DROP_COLS = ["id", "attack_cat"]


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


def create_one_hot_encoder():
    """
    Tương thích nhiều phiên bản scikit-learn.
    Bản mới dùng sparse_output, bản cũ dùng sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()
    print(f"[OK] Saved figure: {FIGURE_DIR / filename}")


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    print_section("1. LOAD RAW DATA")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing testing file: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, test_df


# ============================================================
# 4. VALIDATE DATA
# ============================================================

def validate_data(train_df, test_df):
    print_section("2. VALIDATE DATA")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train data.")

    if TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in test data.")

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols

    if missing_in_test:
        print(f"[WARNING] Columns in train but not in test: {missing_in_test}")

    if missing_in_train:
        print(f"[WARNING] Columns in test but not in train: {missing_in_train}")

    print("\nTarget distribution in train:")
    print(train_df[TARGET_COL].value_counts())

    print("\nTarget distribution in test:")
    print(test_df[TARGET_COL].value_counts())

    label_summary = pd.DataFrame({
        "dataset": ["train", "train", "test", "test"],
        "label": [0, 1, 0, 1],
        "label_name": ["Normal", "Attack", "Normal", "Attack"],
        "count": [
            int((train_df[TARGET_COL] == 0).sum()),
            int((train_df[TARGET_COL] == 1).sum()),
            int((test_df[TARGET_COL] == 0).sum()),
            int((test_df[TARGET_COL] == 1).sum()),
        ],
    })

    label_summary["ratio_percent"] = label_summary.apply(
        lambda row: row["count"] / len(train_df) * 100
        if row["dataset"] == "train"
        else row["count"] / len(test_df) * 100,
        axis=1
    )

    label_summary.to_csv(REPORT_DIR / "preprocessing_label_summary.csv", index=False)


# ============================================================
# 5. CLEAN DATA
# ============================================================

def clean_data(df: pd.DataFrame, dataset_name: str):
    print_section(f"3. CLEAN DATA - {dataset_name.upper()}")

    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    duplicate_count = df.duplicated().sum()
    missing_before = df.isnull().sum().sum()

    print(f"Infinite values before cleaning: {inf_count}")
    print(f"Missing values before cleaning : {missing_before}")
    print(f"Duplicate rows               : {duplicate_count}")

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    missing_after = df.isnull().sum().sum()

    cleaning_summary = {
        "dataset": dataset_name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "infinite_values_before_cleaning": int(inf_count),
        "missing_values_before_cleaning": int(missing_before),
        "missing_values_after_replacing_inf": int(missing_after),
        "duplicate_rows": int(duplicate_count)
    }

    save_json(cleaning_summary, REPORT_DIR / f"cleaning_summary_{dataset_name}.json")

    return df


# ============================================================
# 6. SPLIT FEATURES AND TARGET
# ============================================================

def split_features_target(train_df, test_df):
    print_section("4. SPLIT FEATURES AND TARGET")

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    drop_cols = [col for col in DROP_COLS + [TARGET_COL] if col in train_df.columns]

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    print(f"Dropped columns: {drop_cols}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    split_summary = {
        "target_col": TARGET_COL,
        "dropped_cols": drop_cols,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape
    }

    save_json(split_summary, REPORT_DIR / "split_summary.json")

    return X_train, X_test, y_train, y_test, drop_cols


# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X_train):
    print_section("5. IDENTIFY FEATURE TYPES")

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    other_cols = [
        col for col in X_train.columns
        if col not in categorical_cols and col not in numerical_cols
    ]

    print(f"Categorical columns ({len(categorical_cols)}):")
    print(categorical_cols)

    print(f"\nNumerical columns ({len(numerical_cols)}):")
    print(numerical_cols)

    if other_cols:
        print(f"\nOther columns ({len(other_cols)}):")
        print(other_cols)

    feature_type_summary = {
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "other_cols": other_cols,
        "num_categorical_cols": len(categorical_cols),
        "num_numerical_cols": len(numerical_cols),
        "num_other_cols": len(other_cols)
    }

    save_json(feature_type_summary, REPORT_DIR / "feature_type_summary.json")

    return categorical_cols, numerical_cols, other_cols


# ============================================================
# 8. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(categorical_cols, numerical_cols):
    print_section("6. BUILD PREPROCESSING PIPELINE")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    print("Numerical pipeline:")
    print("- SimpleImputer(strategy='median')")
    print("- StandardScaler()")

    print("\nCategorical pipeline:")
    print("- SimpleImputer(strategy='most_frequent')")
    print("- OneHotEncoder(handle_unknown='ignore')")

    return preprocessor


# ============================================================
# 9. FIT AND TRANSFORM
# ============================================================

def fit_transform_data(preprocessor, X_train, X_test):
    print_section("7. FIT AND TRANSFORM DATA")

    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming testing data...")
    X_test_processed = preprocessor.transform(X_test)

    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape : {X_test_processed.shape}")

    return X_train_processed, X_test_processed


# ============================================================
# 10. GET PROCESSED FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor, categorical_cols, numerical_cols):
    print_section("8. GET PROCESSED FEATURE NAMES")

    feature_names = []

    feature_names.extend(numerical_cols)

    if len(categorical_cols) > 0:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["onehot"]

        try:
            encoded_cat_names = onehot.get_feature_names_out(categorical_cols).tolist()
        except AttributeError:
            encoded_cat_names = onehot.get_feature_names(categorical_cols).tolist()

        feature_names.extend(encoded_cat_names)

    print(f"Number of processed features: {len(feature_names)}")

    return feature_names


# ============================================================
# 11. VISUALIZATION FOR REPORT
# ============================================================

def plot_preprocessing_pipeline_flow():
    print_section("9. VISUALIZE PREPROCESSING PIPELINE")

    plt.figure(figsize=(14, 4))

    steps = [
        "Raw CSV\nUNSW-NB15",
        "Clean data\ninf → NaN",
        "Drop columns\nid, attack_cat",
        "Split X / y\nlabel target",
        "Numerical\nmedian + scaler",
        "Categorical\nmode + one-hot",
        "Processed data\nmodel-ready"
    ]

    x_positions = np.linspace(0.06, 0.94, len(steps))

    for i, (x, step) in enumerate(zip(x_positions, steps)):
        plt.text(
            x,
            0.5,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black")
        )

        if i < len(steps) - 1:
            plt.annotate(
                "",
                xy=(x_positions[i + 1] - 0.055, 0.5),
                xytext=(x + 0.055, 0.5),
                arrowprops=dict(arrowstyle="->")
            )

    plt.axis("off")
    plt.title("Preprocessing Pipeline Flow")
    save_figure("preprocessing_pipeline_flow.png")


def plot_feature_group_counts(categorical_cols, numerical_cols, dropped_cols):
    labels = ["Numerical", "Categorical", "Dropped"]
    values = [len(numerical_cols), len(categorical_cols), len(dropped_cols)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Feature Groups Before Preprocessing")
    plt.xlabel("Feature Group")
    plt.ylabel("Number of Columns")

    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_figure("preprocessing_feature_group_counts.png")

    df = pd.DataFrame({
        "feature_group": labels,
        "count": values
    })
    df.to_csv(REPORT_DIR / "preprocessing_feature_group_counts.csv", index=False)


def plot_feature_count_before_after(X_train, X_train_processed):
    labels = ["Before Preprocessing", "After Preprocessing"]
    values = [X_train.shape[1], X_train_processed.shape[1]]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Feature Count Before and After Preprocessing")
    plt.xlabel("Stage")
    plt.ylabel("Number of Features")

    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_figure("preprocessing_feature_count_before_after.png")

    df = pd.DataFrame({
        "stage": labels,
        "feature_count": values
    })
    df.to_csv(REPORT_DIR / "preprocessing_feature_count_before_after.csv", index=False)


def plot_categorical_cardinality(X_train, categorical_cols):
    if len(categorical_cols) == 0:
        print("No categorical columns to visualize.")
        return

    cardinality_df = pd.DataFrame({
        "column": categorical_cols,
        "unique_count": [X_train[col].nunique() for col in categorical_cols]
    }).sort_values("unique_count", ascending=False)

    cardinality_df.to_csv(REPORT_DIR / "preprocessing_categorical_cardinality.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(cardinality_df["column"], cardinality_df["unique_count"])
    plt.title("Unique Values in Categorical Features")
    plt.xlabel("Categorical Feature")
    plt.ylabel("Number of Unique Values")
    plt.xticks(rotation=45, ha="right")

    for i, value in enumerate(cardinality_df["unique_count"]):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_figure("preprocessing_categorical_cardinality.png")


def plot_onehot_expansion(categorical_cols, feature_names):
    if len(categorical_cols) == 0:
        print("No categorical columns for one-hot expansion.")
        return

    onehot_counts = []

    for col in categorical_cols:
        prefix = f"{col}_"
        count = sum(name.startswith(prefix) for name in feature_names)
        onehot_counts.append({
            "column": col,
            "onehot_output_features": count
        })

    onehot_df = pd.DataFrame(onehot_counts)
    onehot_df.to_csv(REPORT_DIR / "preprocessing_onehot_expansion.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(onehot_df["column"], onehot_df["onehot_output_features"])
    plt.title("One-Hot Encoding Feature Expansion")
    plt.xlabel("Categorical Feature")
    plt.ylabel("Number of Output Features")
    plt.xticks(rotation=45, ha="right")

    for i, value in enumerate(onehot_df["onehot_output_features"]):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_figure("preprocessing_onehot_expansion.png")


def plot_label_distribution(y_train):
    label_counts = pd.Series(y_train).value_counts().sort_index()
    label_names = ["Normal" if label == 0 else "Attack" for label in label_counts.index]

    plt.figure(figsize=(7, 5))
    plt.bar(label_names, label_counts.values)
    plt.title("Label Distribution in Training Set")
    plt.xlabel("Label")
    plt.ylabel("Number of Records")

    for i, value in enumerate(label_counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_figure("preprocessing_label_distribution.png")


def plot_scaling_examples(X_train, X_train_processed, numerical_cols):
    selected_cols = [
        "dur",
        "sbytes",
        "dbytes",
        "sload",
        "dload"
    ]

    selected_cols = [col for col in selected_cols if col in numerical_cols]

    if len(selected_cols) == 0:
        print("No selected numerical columns found for scaling visualization.")
        return

    for col in selected_cols[:3]:
        col_index = numerical_cols.index(col)

        raw_values = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        scaled_values = X_train_processed[:, col_index]

        if sparse.issparse(X_train_processed):
            scaled_values = scaled_values.toarray().ravel()
        else:
            scaled_values = np.asarray(scaled_values).ravel()

        plt.figure(figsize=(8, 5))
        np.log1p(raw_values).hist(bins=50)
        plt.title(f"Before Scaling - log1p({col})")
        plt.xlabel(f"log1p({col})")
        plt.ylabel("Frequency")
        save_figure(f"preprocessing_before_scaling_{col}.png")

        plt.figure(figsize=(8, 5))
        pd.Series(scaled_values).hist(bins=50)
        plt.title(f"After StandardScaler - {col}")
        plt.xlabel(f"scaled {col}")
        plt.ylabel("Frequency")
        save_figure(f"preprocessing_after_scaling_{col}.png")


def create_preprocessing_visualizations(
    X_train,
    X_train_processed,
    y_train,
    categorical_cols,
    numerical_cols,
    dropped_cols,
    feature_names
):
    plot_preprocessing_pipeline_flow()
    plot_feature_group_counts(categorical_cols, numerical_cols, dropped_cols)
    plot_feature_count_before_after(X_train, X_train_processed)
    plot_categorical_cardinality(X_train, categorical_cols)
    plot_onehot_expansion(categorical_cols, feature_names)
    plot_label_distribution(y_train)
    plot_scaling_examples(X_train, X_train_processed, numerical_cols)


# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

def save_outputs(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor,
    feature_names,
    categorical_cols,
    numerical_cols,
    dropped_cols
):
    print_section("10. SAVE PROCESSED DATA AND ARTIFACTS")

    sparse.save_npz(PROCESSED_DIR / "X_train_processed.npz", X_train_processed)
    sparse.save_npz(PROCESSED_DIR / "X_test_processed.npz", X_test_processed)

    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    column_info = {
        "target_col": TARGET_COL,
        "dropped_cols": dropped_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols
    }

    save_json(column_info, PROCESSED_DIR / "column_info.json")
    save_json(feature_names, PROCESSED_DIR / "feature_names.json")

    preprocessing_summary = {
        "target_col": TARGET_COL,
        "dropped_cols": dropped_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "num_original_features": len(categorical_cols) + len(numerical_cols),
        "num_processed_features": len(feature_names),
        "X_train_processed_shape": X_train_processed.shape,
        "X_test_processed_shape": X_test_processed.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "preprocessor_path": str(MODEL_DIR / "preprocessor.joblib")
    }

    save_json(preprocessing_summary, REPORT_DIR / "preprocessing_summary.json")

    print("[OK] Saved processed data:")
    print(f"- {PROCESSED_DIR / 'X_train_processed.npz'}")
    print(f"- {PROCESSED_DIR / 'X_test_processed.npz'}")
    print(f"- {PROCESSED_DIR / 'y_train.npy'}")
    print(f"- {PROCESSED_DIR / 'y_test.npy'}")

    print("\n[OK] Saved artifacts:")
    print(f"- {MODEL_DIR / 'preprocessor.joblib'}")
    print(f"- {PROCESSED_DIR / 'column_info.json'}")
    print(f"- {PROCESSED_DIR / 'feature_names.json'}")
    print(f"- {REPORT_DIR / 'preprocessing_summary.json'}")


# ============================================================
# 13. GENERATE REPORT NOTES
# ============================================================

def generate_report_notes(
    X_train,
    X_train_processed,
    y_train,
    categorical_cols,
    numerical_cols,
    dropped_cols
):
    print_section("11. GENERATE PREPROCESSING REPORT NOTES")

    label_counts = pd.Series(y_train).value_counts().sort_index()

    normal_count = int(label_counts.get(0, 0))
    attack_count = int(label_counts.get(1, 0))

    content = f"""# Preprocessing Report Notes

## Mục tiêu tiền xử lý

Dữ liệu UNSW-NB15 bao gồm cả thuộc tính số và thuộc tính phân loại. Vì các mô hình học máy không thể sử dụng trực tiếp dữ liệu dạng chuỗi, dữ liệu cần được chuyển đổi về dạng số trước khi huấn luyện.

## Các cột bị loại bỏ

Các cột bị loại bỏ:

{dropped_cols}

Trong đó, `id` là mã định danh dòng dữ liệu nên không có ý nghĩa dự đoán. Cột `attack_cat` thể hiện loại tấn công cụ thể, nên không được dùng trong bài toán phân loại nhị phân Normal/Attack để tránh rò rỉ nhãn.

## Tách đặc trưng và nhãn

Cột nhãn sử dụng là `label`, trong đó:

0 = Normal  
1 = Attack

Phân phối nhãn trong tập train:

Normal: {normal_count}  
Attack: {attack_count}

## Xử lý thuộc tính số

Số lượng thuộc tính số: {len(numerical_cols)}

Các thuộc tính số được xử lý bằng:

SimpleImputer(strategy="median")  
StandardScaler()

Median được dùng để điền missing value vì bền hơn mean khi dữ liệu có outlier. StandardScaler giúp đưa các đặc trưng số về cùng thang đo, đặc biệt cần thiết với các mô hình như Logistic Regression.

## Xử lý thuộc tính phân loại

Số lượng thuộc tính phân loại: {len(categorical_cols)}

Các cột phân loại:

{categorical_cols}

Các thuộc tính phân loại được xử lý bằng:

SimpleImputer(strategy="most_frequent")  
OneHotEncoder(handle_unknown="ignore")

OneHotEncoder chuyển các cột dạng chuỗi như `proto`, `service`, `state` thành vector số. Tham số `handle_unknown="ignore"` giúp hệ thống không lỗi nếu dữ liệu test hoặc dữ liệu streaming có category mới.

## Số feature trước và sau tiền xử lý

Số feature trước tiền xử lý: {X_train.shape[1]}  
Số feature sau tiền xử lý: {X_train_processed.shape[1]}

Số feature tăng lên sau tiền xử lý do các thuộc tính phân loại được mở rộng bằng One-Hot Encoding.

## Tránh data leakage

Preprocessor chỉ được fit trên tập train. Tập test chỉ được transform bằng preprocessor đã fit. Cách làm này giúp tránh data leakage, tức là tránh việc thông tin từ tập test bị sử dụng trong quá trình huấn luyện.

## Tái sử dụng trong streaming

Preprocessor được lưu tại:

models/preprocessor.joblib

Trong giai đoạn mô phỏng streaming, các micro-batch mới sẽ phải đi qua đúng preprocessor này trước khi đưa vào mô hình dự đoán.
"""

    report_path = REPORT_DIR / "preprocessing_report_notes.md"
    report_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved report notes: {report_path}")

# ============================================================
# 14. MAIN
# ============================================================

def main():
    train_df, test_df = load_data()

    validate_data(train_df, test_df)

    train_df = clean_data(train_df, "train")
    test_df = clean_data(test_df, "test")

    X_train, X_test, y_train, y_test, dropped_cols = split_features_target(
        train_df,
        test_df
    )

    categorical_cols, numerical_cols, other_cols = identify_feature_types(X_train)

    if other_cols:
        print("[WARNING] Other column types found. They will be ignored.")

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    X_train_processed, X_test_processed = fit_transform_data(
        preprocessor,
        X_train,
        X_test
    )

    feature_names = get_feature_names(
        preprocessor,
        categorical_cols,
        numerical_cols
    )

    create_preprocessing_visualizations(
        X_train=X_train,
        X_train_processed=X_train_processed,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dropped_cols=dropped_cols,
        feature_names=feature_names
    )

    save_outputs(
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dropped_cols=dropped_cols
    )

    generate_report_notes(
        X_train=X_train,
        X_train_processed=X_train_processed,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dropped_cols=dropped_cols
    )

    print("\nPreprocessing completed successfully.")
    print(f"Processed data saved to: {PROCESSED_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()