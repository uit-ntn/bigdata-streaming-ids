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

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "preprocessing"
RESULT_DIR = REPORT_DIR / "results" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

TARGET_COL = "label"

# id: mã định danh dòng dữ liệu, không có ý nghĩa dự đoán.
# attack_cat: loại tấn công cụ thể, không dùng khi train binary Normal/Attack
# để tránh rò rỉ nhãn.
DROP_COLS = ["id", "attack_cat"]


# ============================================================
# 2. UTILITY FUNCTIONS
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


def create_one_hot_encoder():
    """
    Tạo OneHotEncoder tương thích nhiều phiên bản scikit-learn.
    scikit-learn mới dùng sparse_output.
    scikit-learn cũ dùng sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def get_dense_column(matrix, col_index: int):
    """
    Lấy một cột từ sparse/dense matrix và chuyển thành numpy array 1 chiều.
    """
    col_values = matrix[:, col_index]

    if sparse.issparse(matrix):
        return col_values.toarray().ravel()

    return np.asarray(col_values).ravel()


def sample_array(values, max_sample=5000, random_state=42):
    """
    Lấy mẫu để vẽ các biểu đồ như violin/ECDF nhanh hơn.
    """
    values = np.asarray(values)

    if len(values) <= max_sample:
        return values

    rng = np.random.default_rng(random_state)
    return rng.choice(values, size=max_sample, replace=False)


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

def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print_section("2. VALIDATE DATA")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train data.")

    if TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in test data.")

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    missing_in_test = sorted(list(train_cols - test_cols))
    missing_in_train = sorted(list(test_cols - train_cols))

    if missing_in_test:
        print(f"[WARNING] Columns in train but not in test: {missing_in_test}")

    if missing_in_train:
        print(f"[WARNING] Columns in test but not in train: {missing_in_train}")

    train_label_counts = train_df[TARGET_COL].value_counts().sort_index()
    test_label_counts = test_df[TARGET_COL].value_counts().sort_index()

    print("\nTrain label distribution:")
    print(train_label_counts)

    print("\nTest label distribution:")
    print(test_label_counts)

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
        axis=1,
    )

    label_summary.to_csv(RESULT_DIR / "label_summary_before_preprocessing.csv", index=False)

    validation_summary = {
        "target_col": TARGET_COL,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "missing_columns_in_test": missing_in_test,
        "missing_columns_in_train": missing_in_train,
        "train_label_distribution": train_label_counts.to_dict(),
        "test_label_distribution": test_label_counts.to_dict(),
    }

    save_json(validation_summary, RESULT_DIR / "validation_summary.json")

    print("\nValidation completed.")


# ============================================================
# 5. CLEAN DATA
# ============================================================

def clean_data(df: pd.DataFrame, dataset_name: str):
    print_section(f"3. CLEAN DATA - {dataset_name.upper()}")

    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        inf_count = int(np.isinf(df[numeric_cols]).sum().sum())
    else:
        inf_count = 0

    missing_before = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    print(f"Infinite values before cleaning: {inf_count}")
    print(f"Missing values before cleaning : {missing_before}")
    print(f"Duplicate rows                : {duplicate_count}")

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    missing_after = int(df.isnull().sum().sum())

    print(f"Missing values after replacing inf: {missing_after}")

    cleaning_summary = {
        "dataset": dataset_name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "infinite_values_before_cleaning": inf_count,
        "missing_values_before_cleaning": missing_before,
        "missing_values_after_replacing_inf": missing_after,
        "duplicate_rows": duplicate_count,
    }

    save_json(cleaning_summary, RESULT_DIR / f"cleaning_summary_{dataset_name}.json")

    return df


# ============================================================
# 6. SPLIT FEATURES AND TARGET
# ============================================================

def split_features_target(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print_section("4. SPLIT FEATURES AND TARGET")

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    dropped_cols = [col for col in DROP_COLS + [TARGET_COL] if col in train_df.columns]

    X_train = train_df.drop(columns=dropped_cols)
    X_test = test_df.drop(columns=dropped_cols)

    print(f"Dropped columns: {dropped_cols}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    split_summary = {
        "target_col": TARGET_COL,
        "dropped_cols": dropped_cols,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
    }

    save_json(split_summary, RESULT_DIR / "split_summary.json")

    return X_train, X_test, y_train, y_test, dropped_cols


# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X_train: pd.DataFrame):
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
        "num_other_cols": len(other_cols),
    }

    save_json(feature_type_summary, RESULT_DIR / "feature_type_summary.json")

    return categorical_cols, numerical_cols, other_cols


# ============================================================
# 8. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(categorical_cols, numerical_cols):
    print_section("6. BUILD PREPROCESSING PIPELINE")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    print("Numerical pipeline:")
    print("- SimpleImputer(strategy='median')")
    print("- StandardScaler()")

    print("\nCategorical pipeline:")
    print("- SimpleImputer(strategy='most_frequent')")
    print("- OneHotEncoder(handle_unknown='ignore')")

    pipeline_summary = {
        "numerical_pipeline": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()",
        ],
        "categorical_pipeline": [
            "SimpleImputer(strategy='most_frequent')",
            "OneHotEncoder(handle_unknown='ignore')",
        ],
    }

    save_json(pipeline_summary, RESULT_DIR / "preprocessing_pipeline_summary.json")

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

    if categorical_cols:
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
    plt.figure(figsize=(15, 4))

    steps = [
        "Raw CSV\nUNSW-NB15",
        "Clean data\ninf -> NaN",
        "Drop columns\nid, attack_cat",
        "Split X / y\nlabel target",
        "Numerical\nmedian + scaler",
        "Categorical\nmode + one-hot",
        "Processed data\nmodel-ready",
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
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black"),
        )

        if i < len(steps) - 1:
            plt.annotate(
                "",
                xy=(x_positions[i + 1] - 0.055, 0.5),
                xytext=(x + 0.055, 0.5),
                arrowprops=dict(arrowstyle="->"),
            )

    plt.axis("off")
    plt.title("Preprocessing Pipeline Flow")
    save_figure("01_preprocessing_pipeline_flow.png")


def plot_feature_group_donut(categorical_cols, numerical_cols, dropped_cols):
    labels = ["Numerical", "Categorical", "Dropped"]
    values = [len(numerical_cols), len(categorical_cols), len(dropped_cols)]

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
        f"{sum(values)}\ncolumns",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.title("Feature Groups Before Preprocessing")
    save_figure("02_feature_group_donut.png")

    df = pd.DataFrame({
        "feature_group": labels,
        "count": values,
    })
    df.to_csv(RESULT_DIR / "feature_group_counts.csv", index=False)


def plot_feature_count_slope(X_train, X_train_processed):
    before_count = X_train.shape[1]
    after_count = X_train_processed.shape[1]

    plt.figure(figsize=(7, 5))

    x = [0, 1]
    y = [before_count, after_count]

    plt.plot(x, y, marker="o", linewidth=2)

    plt.text(0, before_count, f"Before\n{before_count}", ha="right", va="center")
    plt.text(1, after_count, f"After\n{after_count}", ha="left", va="center")

    plt.xticks([0, 1], ["Before", "After"])
    plt.ylabel("Number of Features")
    plt.title("Feature Expansion After Preprocessing")
    plt.grid(axis="y", alpha=0.3)

    save_figure("03_feature_count_slope_chart.png")

    df = pd.DataFrame({
        "stage": ["before_preprocessing", "after_preprocessing"],
        "feature_count": [before_count, after_count],
    })
    df.to_csv(RESULT_DIR / "feature_count_before_after.csv", index=False)


def plot_categorical_cardinality_lollipop(X_train, categorical_cols):
    if not categorical_cols:
        print("No categorical columns to visualize.")
        return

    cardinality_df = pd.DataFrame({
        "column": categorical_cols,
        "unique_count": [X_train[col].nunique() for col in categorical_cols],
    }).sort_values("unique_count", ascending=True)

    cardinality_df.to_csv(RESULT_DIR / "categorical_cardinality.csv", index=False)

    plt.figure(figsize=(8, 5))

    y_pos = np.arange(len(cardinality_df))

    plt.hlines(
        y=y_pos,
        xmin=0,
        xmax=cardinality_df["unique_count"],
        linewidth=2,
    )
    plt.plot(cardinality_df["unique_count"], y_pos, "o")

    plt.yticks(y_pos, cardinality_df["column"])
    plt.xlabel("Number of Unique Values")
    plt.ylabel("Categorical Feature")
    plt.title("Categorical Feature Cardinality")

    for i, value in enumerate(cardinality_df["unique_count"]):
        plt.text(value, i, f" {value}", va="center")

    save_figure("04_categorical_cardinality_lollipop.png")


def plot_onehot_expansion_lollipop(categorical_cols, feature_names):
    if not categorical_cols:
        print("No categorical columns for one-hot expansion.")
        return

    onehot_counts = []

    for col in categorical_cols:
        prefix = f"{col}_"
        count = sum(name.startswith(prefix) for name in feature_names)

        onehot_counts.append({
            "column": col,
            "onehot_output_features": count,
        })

    onehot_df = pd.DataFrame(onehot_counts).sort_values(
        "onehot_output_features",
        ascending=True,
    )

    onehot_df.to_csv(RESULT_DIR / "onehot_expansion.csv", index=False)

    plt.figure(figsize=(8, 5))

    y_pos = np.arange(len(onehot_df))

    plt.hlines(
        y=y_pos,
        xmin=0,
        xmax=onehot_df["onehot_output_features"],
        linewidth=2,
    )
    plt.plot(onehot_df["onehot_output_features"], y_pos, "o")

    plt.yticks(y_pos, onehot_df["column"])
    plt.xlabel("Number of Output Features")
    plt.ylabel("Categorical Feature")
    plt.title("One-Hot Encoding Expansion")

    for i, value in enumerate(onehot_df["onehot_output_features"]):
        plt.text(value, i, f" {value}", va="center")

    save_figure("05_onehot_expansion_lollipop.png")


def plot_label_distribution_donut(y_train):
    label_counts = pd.Series(y_train).value_counts().sort_index()
    label_names = ["Normal" if label == 0 else "Attack" for label in label_counts.index]

    plt.figure(figsize=(7, 6))
    plt.pie(
        label_counts.values,
        labels=label_names,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.42},
    )

    plt.text(
        0,
        0,
        f"{len(y_train):,}\nrecords",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.title("Label Distribution in Training Set")
    save_figure("06_label_distribution_donut.png")

    label_df = pd.DataFrame({
        "label": label_counts.index,
        "label_name": label_names,
        "count": label_counts.values,
        "ratio_percent": label_counts.values / len(y_train) * 100,
    })

    label_df.to_csv(RESULT_DIR / "label_distribution_after_split.csv", index=False)


def plot_missing_values_matrix(train_df, test_df):
    train_missing = train_df.isnull().astype(int)
    test_missing = test_df.isnull().astype(int)

    sample_train = train_missing.head(300)
    sample_test = test_missing.head(300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(sample_train.T, aspect="auto", interpolation="nearest")
    axes[0].set_title("Missing Value Matrix - Train Sample")
    axes[0].set_xlabel("Row Sample")
    axes[0].set_ylabel("Columns")
    axes[0].set_yticks(range(len(train_df.columns)))
    axes[0].set_yticklabels(train_df.columns, fontsize=6)

    axes[1].imshow(sample_test.T, aspect="auto", interpolation="nearest")
    axes[1].set_title("Missing Value Matrix - Test Sample")
    axes[1].set_xlabel("Row Sample")
    axes[1].set_ylabel("Columns")
    axes[1].set_yticks(range(len(test_df.columns)))
    axes[1].set_yticklabels(test_df.columns, fontsize=6)

    save_path = FIGURE_DIR / "07_missing_value_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[OK] Saved figure: {save_path}")

    summary_df = pd.DataFrame({
        "dataset": ["train", "test"],
        "missing_values": [
            int(train_df.isnull().sum().sum()),
            int(test_df.isnull().sum().sum()),
        ],
    })
    summary_df.to_csv(RESULT_DIR / "missing_values_before_imputation.csv", index=False)


def plot_missing_values_text_card(train_df, test_df):
    train_missing = int(train_df.isnull().sum().sum())
    test_missing = int(test_df.isnull().sum().sum())

    plt.figure(figsize=(8, 4))
    plt.axis("off")

    text = (
        "Missing Values Before Imputation\n\n"
        f"Train missing values: {train_missing:,}\n"
        f"Test missing values : {test_missing:,}\n\n"
        "Missing values and infinite values are handled by the preprocessing pipeline."
    )

    plt.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="black"),
    )

    save_figure("08_missing_values_text_card.png")


def plot_scaling_histograms(X_train, X_train_processed, numerical_cols):
    selected_cols = [
        "dur",
        "sbytes",
        "dbytes",
        "sload",
        "dload",
    ]

    selected_cols = [col for col in selected_cols if col in numerical_cols]

    if not selected_cols:
        print("No selected numerical columns found for scaling visualization.")
        return

    for col in selected_cols[:3]:
        col_index = numerical_cols.index(col)

        raw_values = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        raw_values = raw_values[raw_values >= 0]

        scaled_values = get_dense_column(X_train_processed, col_index)

        plt.figure(figsize=(8, 5))
        plt.hist(np.log1p(raw_values), bins=50)
        plt.title(f"Before Scaling - log1p({col})")
        plt.xlabel(f"log1p({col})")
        plt.ylabel("Frequency")
        save_figure(f"09_hist_before_scaling_{col}.png")

        plt.figure(figsize=(8, 5))
        plt.hist(scaled_values, bins=50)
        plt.title(f"After StandardScaler - {col}")
        plt.xlabel(f"scaled {col}")
        plt.ylabel("Frequency")
        save_figure(f"10_hist_after_scaling_{col}.png")


def plot_scaling_density_lines(X_train, X_train_processed, numerical_cols):
    selected_cols = ["dur", "sbytes", "dbytes"]
    selected_cols = [col for col in selected_cols if col in numerical_cols]

    for col in selected_cols:
        col_index = numerical_cols.index(col)

        raw_values = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        raw_values = raw_values[raw_values >= 0]
        raw_values = np.log1p(raw_values)

        scaled_values = get_dense_column(X_train_processed, col_index)

        raw_density, raw_bins = np.histogram(raw_values, bins=60, density=True)
        scaled_density, scaled_bins = np.histogram(scaled_values, bins=60, density=True)

        raw_centers = (raw_bins[:-1] + raw_bins[1:]) / 2
        scaled_centers = (scaled_bins[:-1] + scaled_bins[1:]) / 2

        plt.figure(figsize=(8, 5))
        plt.plot(raw_centers, raw_density, label=f"Before: log1p({col})")
        plt.plot(scaled_centers, scaled_density, label=f"After: scaled {col}")

        plt.title(f"Density Line Before and After Scaling - {col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        save_figure(f"11_density_before_after_scaling_{col}.png")


def plot_scaling_violin(X_train, X_train_processed, numerical_cols):
    selected_cols = ["dur", "sbytes", "dbytes"]
    selected_cols = [col for col in selected_cols if col in numerical_cols]

    for col in selected_cols:
        col_index = numerical_cols.index(col)

        raw_values = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        raw_values = raw_values[raw_values >= 0]
        raw_values = np.log1p(raw_values).values

        scaled_values = get_dense_column(X_train_processed, col_index)

        raw_values = sample_array(raw_values, max_sample=5000)
        scaled_values = sample_array(scaled_values, max_sample=5000)

        plt.figure(figsize=(7, 5))
        plt.violinplot(
            [raw_values, scaled_values],
            showmeans=True,
            showmedians=True,
        )
        plt.xticks([1, 2], [f"log1p({col})", f"scaled {col}"])
        plt.ylabel("Value")
        plt.title(f"Violin Plot Before and After Scaling - {col}")

        save_figure(f"12_violin_before_after_scaling_{col}.png")


def plot_scaling_ecdf(X_train, X_train_processed, numerical_cols):
    selected_cols = ["dur", "sbytes", "dbytes"]
    selected_cols = [col for col in selected_cols if col in numerical_cols]

    for col in selected_cols:
        col_index = numerical_cols.index(col)

        raw_values = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        raw_values = raw_values[raw_values >= 0]
        raw_values = np.log1p(raw_values).values

        scaled_values = get_dense_column(X_train_processed, col_index)

        raw_values = sample_array(raw_values, max_sample=6000)
        scaled_values = sample_array(scaled_values, max_sample=6000)

        raw_sorted = np.sort(raw_values)
        scaled_sorted = np.sort(scaled_values)

        raw_y = np.arange(1, len(raw_sorted) + 1) / len(raw_sorted)
        scaled_y = np.arange(1, len(scaled_sorted) + 1) / len(scaled_sorted)

        plt.figure(figsize=(8, 5))
        plt.plot(raw_sorted, raw_y, label=f"Before: log1p({col})")
        plt.plot(scaled_sorted, scaled_y, label=f"After: scaled {col}")

        plt.title(f"ECDF Before and After Scaling - {col}")
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        plt.legend()

        save_figure(f"13_ecdf_before_after_scaling_{col}.png")


def plot_processed_matrix_sparsity(X_train_processed):
    if sparse.issparse(X_train_processed):
        total_values = X_train_processed.shape[0] * X_train_processed.shape[1]
        nonzero_values = X_train_processed.nnz
    else:
        total_values = X_train_processed.size
        nonzero_values = np.count_nonzero(X_train_processed)

    zero_values = total_values - nonzero_values

    labels = ["Zero values", "Non-zero values"]
    values = [zero_values, nonzero_values]

    plt.figure(figsize=(7, 6))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops={"width": 0.42},
    )

    plt.title("Sparsity of Processed Feature Matrix")

    save_figure("14_processed_matrix_sparsity_donut.png")

    sparsity_df = pd.DataFrame({
        "type": labels,
        "count": values,
        "ratio_percent": [value / total_values * 100 for value in values],
    })

    sparsity_df.to_csv(RESULT_DIR / "processed_matrix_sparsity.csv", index=False)


def plot_preprocessing_summary_table(
    X_train,
    X_train_processed,
    categorical_cols,
    numerical_cols,
    dropped_cols,
):
    summary_rows = [
        ["Original features", X_train.shape[1]],
        ["Processed features", X_train_processed.shape[1]],
        ["Numerical columns", len(numerical_cols)],
        ["Categorical columns", len(categorical_cols)],
        ["Dropped columns", len(dropped_cols)],
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    table = ax.table(
        cellText=summary_rows,
        colLabels=["Item", "Value"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.title("Preprocessing Summary Table")

    save_path = FIGURE_DIR / "15_preprocessing_summary_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {save_path}")


def create_preprocessing_visualizations(
    train_df,
    test_df,
    X_train,
    X_train_processed,
    y_train,
    categorical_cols,
    numerical_cols,
    dropped_cols,
    feature_names,
):
    print_section("9. CREATE PREPROCESSING VISUALIZATIONS")

    plot_preprocessing_pipeline_flow()
    plot_feature_group_donut(categorical_cols, numerical_cols, dropped_cols)
    plot_feature_count_slope(X_train, X_train_processed)
    plot_categorical_cardinality_lollipop(X_train, categorical_cols)
    plot_onehot_expansion_lollipop(categorical_cols, feature_names)
    plot_label_distribution_donut(y_train)
    plot_missing_values_matrix(train_df, test_df)
    plot_missing_values_text_card(train_df, test_df)
    plot_scaling_histograms(X_train, X_train_processed, numerical_cols)
    plot_scaling_density_lines(X_train, X_train_processed, numerical_cols)
    plot_scaling_violin(X_train, X_train_processed, numerical_cols)
    plot_scaling_ecdf(X_train, X_train_processed, numerical_cols)
    plot_processed_matrix_sparsity(X_train_processed)
    plot_preprocessing_summary_table(
        X_train,
        X_train_processed,
        categorical_cols,
        numerical_cols,
        dropped_cols,
    )


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
    dropped_cols,
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
        "numerical_cols": numerical_cols,
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
        "preprocessor_path": str(MODEL_DIR / "preprocessor.joblib"),
    }

    save_json(preprocessing_summary, RESULT_DIR / "preprocessing_summary.json")

    print("[OK] Saved processed data:")
    print(f"- {PROCESSED_DIR / 'X_train_processed.npz'}")
    print(f"- {PROCESSED_DIR / 'X_test_processed.npz'}")
    print(f"- {PROCESSED_DIR / 'y_train.npy'}")
    print(f"- {PROCESSED_DIR / 'y_test.npy'}")

    print("\n[OK] Saved artifacts:")
    print(f"- {MODEL_DIR / 'preprocessor.joblib'}")
    print(f"- {PROCESSED_DIR / 'column_info.json'}")
    print(f"- {PROCESSED_DIR / 'feature_names.json'}")
    print(f"- {RESULT_DIR / 'preprocessing_summary.json'}")


# ============================================================
# 13. GENERATE REPORT NOTES
# ============================================================

def generate_report_notes(
    X_train,
    X_train_processed,
    y_train,
    categorical_cols,
    numerical_cols,
    dropped_cols,
):
    print_section("11. GENERATE PREPROCESSING REPORT NOTES")

    label_counts = pd.Series(y_train).value_counts().sort_index()

    normal_count = int(label_counts.get(0, 0))
    attack_count = int(label_counts.get(1, 0))

    lines = [
        "# Preprocessing Report Notes",
        "",
        "## 1. Mục tiêu tiền xử lý",
        "",
        "Dữ liệu UNSW-NB15 bao gồm cả thuộc tính số và thuộc tính phân loại.",
        "Vì các mô hình học máy không thể sử dụng trực tiếp dữ liệu dạng chuỗi,",
        "dữ liệu cần được chuyển đổi về dạng số trước khi huấn luyện.",
        "",
        "## 2. Các cột bị loại bỏ",
        "",
        f"Các cột bị loại bỏ: {dropped_cols}",
        "",
        "`id` là mã định danh dòng dữ liệu nên không có ý nghĩa dự đoán.",
        "`attack_cat` thể hiện loại tấn công cụ thể, nên không được dùng trong bài toán phân loại nhị phân Normal/Attack để tránh rò rỉ nhãn.",
        "",
        "## 3. Tách đặc trưng và nhãn",
        "",
        "Cột nhãn sử dụng là `label`:",
        "",
        "- 0 = Normal",
        "- 1 = Attack",
        "",
        "Phân phối nhãn trong tập train:",
        "",
        f"- Normal: {normal_count:,}",
        f"- Attack: {attack_count:,}",
        "",
        "## 4. Xử lý thuộc tính số",
        "",
        f"Số lượng thuộc tính số: {len(numerical_cols)}",
        "",
        "Các thuộc tính số được xử lý bằng:",
        "",
        "- SimpleImputer(strategy='median')",
        "- StandardScaler()",
        "",
        "Median được dùng để điền missing value vì bền hơn mean khi dữ liệu có outlier.",
        "StandardScaler giúp đưa các đặc trưng số về cùng thang đo, đặc biệt cần thiết với các mô hình như Logistic Regression.",
        "",
        "## 5. Xử lý thuộc tính phân loại",
        "",
        f"Số lượng thuộc tính phân loại: {len(categorical_cols)}",
        f"Các cột phân loại: {categorical_cols}",
        "",
        "Các thuộc tính phân loại được xử lý bằng:",
        "",
        "- SimpleImputer(strategy='most_frequent')",
        "- OneHotEncoder(handle_unknown='ignore')",
        "",
        "OneHotEncoder chuyển các cột dạng chuỗi như `proto`, `service`, `state` thành vector số.",
        "Tham số `handle_unknown='ignore'` giúp hệ thống không lỗi nếu dữ liệu test hoặc dữ liệu streaming có category mới.",
        "",
        "## 6. Số feature trước và sau tiền xử lý",
        "",
        f"Số feature trước tiền xử lý: {X_train.shape[1]}",
        f"Số feature sau tiền xử lý: {X_train_processed.shape[1]}",
        "",
        "Số feature tăng lên sau tiền xử lý do các thuộc tính phân loại được mở rộng bằng One-Hot Encoding.",
        "",
        "## 7. Tránh data leakage",
        "",
        "Preprocessor chỉ được fit trên tập train.",
        "Tập test chỉ được transform bằng preprocessor đã fit.",
        "Cách làm này giúp tránh data leakage, tức là tránh việc thông tin từ tập test bị sử dụng trong quá trình huấn luyện.",
        "",
        "## 8. Tái sử dụng trong streaming",
        "",
        "Preprocessor được lưu tại:",
        "",
        "models/preprocessor.joblib",
        "",
        "Trong giai đoạn mô phỏng streaming, các micro-batch mới sẽ phải đi qua đúng preprocessor này trước khi đưa vào mô hình dự đoán.",
        "",
        "## 9. Output",
        "",
        "Dữ liệu processed được lưu trong:",
        "",
        "data/processed/",
        "",
        "Biểu đồ preprocessing được lưu trong:",
        "",
        "reports/figures/preprocessing/",
        "",
        "Bảng kết quả preprocessing được lưu trong:",
        "",
        "reports/results/preprocessing/",
        "",
        "## 10. Các loại biểu đồ đã tạo",
        "",
        "- Flow diagram: mô tả luồng tiền xử lý.",
        "- Donut chart: tỷ trọng nhóm feature và độ thưa của ma trận sau xử lý.",
        "- Slope chart: số feature trước và sau tiền xử lý.",
        "- Lollipop chart: cardinality và số feature sau One-Hot Encoding.",
        "- Missing-value matrix: trực quan tình trạng missing value.",
        "- Histogram, density line, violin plot, ECDF: so sánh dữ liệu số trước và sau scaling.",
        "- Summary table image: bảng tóm tắt tiền xử lý.",
    ]

    content = "\n".join(lines)

    report_path = RESULT_DIR / "preprocessing_report_notes.md"
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
        test_df,
    )

    categorical_cols, numerical_cols, other_cols = identify_feature_types(X_train)

    if other_cols:
        print("[WARNING] Other column types found. They will be ignored by current pipeline.")

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    X_train_processed, X_test_processed = fit_transform_data(
        preprocessor,
        X_train,
        X_test,
    )

    feature_names = get_feature_names(
        preprocessor,
        categorical_cols,
        numerical_cols,
    )

    create_preprocessing_visualizations(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_train_processed=X_train_processed,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dropped_cols=dropped_cols,
        feature_names=feature_names,
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
        dropped_cols=dropped_cols,
    )

    generate_report_notes(
        X_train=X_train,
        X_train_processed=X_train_processed,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        dropped_cols=dropped_cols,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Processed data saved to: {PROCESSED_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Reports saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()