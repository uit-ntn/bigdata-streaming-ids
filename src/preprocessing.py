from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
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

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

TARGET_COL = "label"

# id: chỉ là mã dòng, không có ý nghĩa dự đoán
# attack_cat: loại tấn công, không dùng khi train binary label
DROP_COLS = ["id", "attack_cat"]


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


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

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    return train_df, test_df


# ============================================================
# 4. DATA VALIDATION
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

    print("\nValidation completed.")


# ============================================================
# 5. CLEAN DATA
# ============================================================

def clean_data(df: pd.DataFrame, dataset_name: str):
    print_section(f"3. CLEAN DATA - {dataset_name.upper()}")

    df = df.copy()

    original_shape = df.shape

    # Xử lý giá trị vô cực nếu có
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Number of infinite values before cleaning: {inf_count}")

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Không drop missing ở đây, vì lát nữa SimpleImputer sẽ xử lý
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values after replacing inf: {missing_count}")

    # Kiểm tra duplicate nhưng chưa xóa tự động
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape : {df.shape}")

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

    return X_train, X_test, y_train, y_test, drop_cols


# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X_train):
    print_section("5. IDENTIFY FEATURE TYPES")

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Những cột không phải object nhưng cũng không phải number, nếu có
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
            # Median bền hơn mean nếu dữ liệu lệch hoặc có outlier
            ("imputer", SimpleImputer(strategy="median")),

            # StandardScaler cần cho Logistic Regression, SVM, KNN...
            # Với tree-based models thì không bắt buộc, nhưng vẫn dùng chung pipeline cho nhất quán
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Nếu cột categorical bị thiếu, điền bằng giá trị xuất hiện nhiều nhất
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # OneHotEncoder biến proto/service/state thành vector số
            # handle_unknown="ignore" giúp test có category mới không bị lỗi
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

    print("Preprocessor created.")
    print("Numerical pipeline: median imputer + standard scaler")
    print("Categorical pipeline: most frequent imputer + one-hot encoder")

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
# 10. GET OUTPUT FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor, categorical_cols, numerical_cols):
    print_section("8. GET PROCESSED FEATURE NAMES")

    feature_names = []

    # Numerical feature names giữ nguyên
    feature_names.extend(numerical_cols)

    # Categorical feature names sau one-hot
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
# 11. SAVE OUTPUTS
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
    print_section("9. SAVE PROCESSED DATA AND ARTIFACTS")

    # Lưu sparse matrix để tiết kiệm dung lượng vì OneHotEncoder tạo nhiều cột 0
    sparse.save_npz(PROCESSED_DIR / "X_train_processed.npz", X_train_processed)
    sparse.save_npz(PROCESSED_DIR / "X_test_processed.npz", X_test_processed)

    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    # Lưu preprocessor để sau này dùng lại trong streaming simulation
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

    print("\n[OK] Saved preprocessing artifacts:")
    print(f"- {MODEL_DIR / 'preprocessor.joblib'}")
    print(f"- {PROCESSED_DIR / 'column_info.json'}")
    print(f"- {PROCESSED_DIR / 'feature_names.json'}")
    print(f"- {REPORT_DIR / 'preprocessing_summary.json'}")


# ============================================================
# 12. MAIN
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
        print("[WARNING] Other column types found. They will be ignored by current pipeline.")

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

    save_outputs(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names,
        categorical_cols,
        numerical_cols,
        dropped_cols
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()