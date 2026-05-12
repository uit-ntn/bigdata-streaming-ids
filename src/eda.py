from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RESULT_DIR = REPORT_DIR / "results"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"


def load_data():
    print("Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("[OK] Dataset loaded.")
    return train_df, test_df


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def basic_overview(train_df, test_df):
    print_section("1. BASIC OVERVIEW")

    overview = pd.DataFrame({
        "dataset": ["train", "test"],
        "rows": [train_df.shape[0], test_df.shape[0]],
        "columns": [train_df.shape[1], test_df.shape[1]],
    })

    print(overview)
    overview.to_csv(RESULT_DIR / "dataset_overview.csv", index=False)

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    columns_df = pd.DataFrame({
        "column": train_df.columns,
        "dtype": train_df.dtypes.astype(str).values,
        "non_null_count": train_df.notnull().sum().values,
        "null_count": train_df.isnull().sum().values,
        "unique_count": train_df.nunique().values,
    })

    columns_df.to_csv(RESULT_DIR / "column_summary.csv", index=False)

    print("\nColumn summary saved to reports/results/column_summary.csv")


def missing_value_analysis(train_df, test_df):
    print_section("2. MISSING VALUE ANALYSIS")

    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()

    missing_df = pd.DataFrame({
        "column": train_df.columns,
        "train_missing": train_missing.values,
        "train_missing_ratio": (train_missing.values / len(train_df)) * 100,
        "test_missing": test_missing.reindex(train_df.columns, fill_value=0).values,
        "test_missing_ratio": (test_missing.reindex(train_df.columns, fill_value=0).values / len(test_df)) * 100,
    })

    missing_df = missing_df[
        (missing_df["train_missing"] > 0) | (missing_df["test_missing"] > 0)
    ]

    if missing_df.empty:
        print("No missing values found in train/test.")
    else:
        print(missing_df)
        missing_df.to_csv(RESULT_DIR / "missing_values.csv", index=False)
        print("Missing value report saved to reports/results/missing_values.csv")


def duplicate_analysis(train_df, test_df):
    print_section("3. DUPLICATE ANALYSIS")

    train_duplicates = train_df.duplicated().sum()
    test_duplicates = test_df.duplicated().sum()

    duplicate_df = pd.DataFrame({
        "dataset": ["train", "test"],
        "duplicate_rows": [train_duplicates, test_duplicates],
        "duplicate_ratio": [
            train_duplicates / len(train_df) * 100,
            test_duplicates / len(test_df) * 100,
        ]
    })

    print(duplicate_df)
    duplicate_df.to_csv(RESULT_DIR / "duplicate_summary.csv", index=False)


def label_distribution(train_df, test_df):
    print_section("4. LABEL DISTRIBUTION")

    train_counts = train_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()

    label_df = pd.DataFrame({
        "label": train_counts.index,
        "train_count": train_counts.values,
        "train_ratio": train_counts.values / len(train_df) * 100,
        "test_count": test_counts.reindex(train_counts.index, fill_value=0).values,
        "test_ratio": test_counts.reindex(train_counts.index, fill_value=0).values / len(test_df) * 100,
    })

    label_df["label_name"] = label_df["label"].map({
        0: "Normal",
        1: "Attack"
    })

    print(label_df)
    label_df.to_csv(RESULT_DIR / "label_distribution.csv", index=False)

    plt.figure(figsize=(7, 5))
    train_counts.plot(kind="bar")
    plt.title("Train Label Distribution")
    plt.xlabel("Label: 0 = Normal, 1 = Attack")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution_train.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    label_df.set_index("label_name")[["train_count", "test_count"]].plot(kind="bar")
    plt.title("Train vs Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution_train_test.png")
    plt.close()


def attack_category_distribution(train_df, test_df):
    print_section("5. ATTACK CATEGORY DISTRIBUTION")

    if "attack_cat" not in train_df.columns:
        print("Column attack_cat not found.")
        return

    train_attack = train_df["attack_cat"].value_counts()
    test_attack = test_df["attack_cat"].value_counts()

    all_categories = sorted(set(train_attack.index).union(set(test_attack.index)))

    attack_df = pd.DataFrame({
        "attack_cat": all_categories,
        "train_count": [train_attack.get(cat, 0) for cat in all_categories],
        "test_count": [test_attack.get(cat, 0) for cat in all_categories],
    })

    attack_df["train_ratio"] = attack_df["train_count"] / len(train_df) * 100
    attack_df["test_ratio"] = attack_df["test_count"] / len(test_df) * 100

    attack_df = attack_df.sort_values("train_count", ascending=False)

    print(attack_df)
    attack_df.to_csv(RESULT_DIR / "attack_category_distribution.csv", index=False)

    plt.figure(figsize=(11, 6))
    train_attack.plot(kind="bar")
    plt.title("Attack Category Distribution in Training Set")
    plt.xlabel("Attack Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "attack_category_distribution_train.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    attack_df.set_index("attack_cat")[["train_count", "test_count"]].plot(kind="bar")
    plt.title("Train vs Test Attack Category Distribution")
    plt.xlabel("Attack Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "attack_category_distribution_train_test.png")
    plt.close()


def categorical_feature_analysis(train_df):
    print_section("6. CATEGORICAL FEATURE ANALYSIS")

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Categorical columns ({len(categorical_cols)}):")
    print(categorical_cols)

    categorical_summary = []

    for col in categorical_cols:
        unique_count = train_df[col].nunique()
        top_value = train_df[col].value_counts().index[0]
        top_count = train_df[col].value_counts().iloc[0]
        top_ratio = top_count / len(train_df) * 100

        categorical_summary.append({
            "column": col,
            "unique_count": unique_count,
            "top_value": top_value,
            "top_count": top_count,
            "top_ratio": top_ratio,
        })

        print(f"\nColumn: {col}")
        print(train_df[col].value_counts().head(15))

        plt.figure(figsize=(10, 5))
        train_df[col].value_counts().head(15).plot(kind="bar")
        plt.title(f"Top 15 Values of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"top_values_{col}.png")
        plt.close()

    categorical_summary_df = pd.DataFrame(categorical_summary)
    categorical_summary_df.to_csv(RESULT_DIR / "categorical_summary.csv", index=False)


def numerical_feature_analysis(train_df):
    print_section("7. NUMERICAL FEATURE ANALYSIS")

    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Numerical columns ({len(numerical_cols)}):")
    print(numerical_cols)

    numerical_summary = train_df[numerical_cols].describe().T
    numerical_summary["missing"] = train_df[numerical_cols].isnull().sum()
    numerical_summary["unique_count"] = train_df[numerical_cols].nunique()

    numerical_summary.to_csv(RESULT_DIR / "numerical_summary.csv")

    print("\nNumerical summary saved to reports/results/numerical_summary.csv")
    print(numerical_summary.head(20))

    selected_cols = [
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "smean",
        "dmean",
    ]

    selected_cols = [col for col in selected_cols if col in train_df.columns]

    for col in selected_cols:
        plt.figure(figsize=(8, 5))
        train_df[col].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"distribution_{col}.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        np.log1p(train_df[col].replace([np.inf, -np.inf], np.nan).dropna()).hist(bins=50)
        plt.title(f"Log Distribution of {col}")
        plt.xlabel(f"log1p({col})")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"log_distribution_{col}.png")
        plt.close()


def feature_by_label_analysis(train_df):
    print_section("8. FEATURE BY LABEL ANALYSIS")

    if "label" not in train_df.columns:
        print("Column label not found.")
        return

    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    ignored_cols = ["id", "label"]
    numerical_cols = [col for col in numerical_cols if col not in ignored_cols]

    group_summary = train_df.groupby("label")[numerical_cols].mean().T
    group_summary.columns = ["normal_mean" if col == 0 else "attack_mean" for col in group_summary.columns]

    if "normal_mean" in group_summary.columns and "attack_mean" in group_summary.columns:
        group_summary["difference"] = group_summary["attack_mean"] - group_summary["normal_mean"]
        group_summary["abs_difference"] = group_summary["difference"].abs()
        group_summary = group_summary.sort_values("abs_difference", ascending=False)

    group_summary.to_csv(RESULT_DIR / "feature_mean_by_label.csv")

    print("Top features with largest mean difference between Normal and Attack:")
    print(group_summary.head(15))

    top_features = group_summary.head(10).index.tolist()

    for col in top_features:
        plt.figure(figsize=(8, 5))
        train_df.boxplot(column=col, by="label")
        plt.title(f"{col} by Label")
        plt.suptitle("")
        plt.xlabel("Label: 0 = Normal, 1 = Attack")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"boxplot_{col}_by_label.png")
        plt.close()


def correlation_analysis(train_df):
    print_section("9. CORRELATION ANALYSIS")

    numerical_df = train_df.select_dtypes(include=[np.number]).copy()

    if "id" in numerical_df.columns:
        numerical_df = numerical_df.drop(columns=["id"])

    corr = numerical_df.corr()

    corr.to_csv(RESULT_DIR / "correlation_matrix.csv")

    if "label" in corr.columns:
        label_corr = corr["label"].drop("label").sort_values(key=lambda x: x.abs(), ascending=False)
        label_corr.to_csv(RESULT_DIR / "label_correlation.csv")

        print("Top features correlated with label:")
        print(label_corr.head(20))

        plt.figure(figsize=(10, 6))
        label_corr.head(20).sort_values().plot(kind="barh")
        plt.title("Top 20 Features Correlated with Label")
        plt.xlabel("Correlation with Label")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "top_label_correlation.png")
        plt.close()

    top_corr_cols = numerical_df.var().sort_values(ascending=False).head(15).index.tolist()
    corr_subset = numerical_df[top_corr_cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_subset, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(top_corr_cols)), top_corr_cols, rotation=90)
    plt.yticks(range(len(top_corr_cols)), top_corr_cols)
    plt.title("Correlation Heatmap of Selected Numerical Features")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap_selected_features.png")
    plt.close()


def train_test_difference_analysis(train_df, test_df):
    print_section("10. TRAIN/TEST DIFFERENCE ANALYSIS")

    common_cols = [col for col in train_df.columns if col in test_df.columns]

    diff_summary = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_mean = train_df[col].mean()
            test_mean = test_df[col].mean()
            train_std = train_df[col].std()
            test_std = test_df[col].std()

            diff_summary.append({
                "column": col,
                "type": "numerical",
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_diff": test_mean - train_mean,
                "train_std": train_std,
                "test_std": test_std,
            })
        else:
            train_unique = train_df[col].nunique()
            test_unique = test_df[col].nunique()

            diff_summary.append({
                "column": col,
                "type": "categorical",
                "train_unique": train_unique,
                "test_unique": test_unique,
            })

    diff_df = pd.DataFrame(diff_summary)
    diff_df.to_csv(RESULT_DIR / "train_test_difference_summary.csv", index=False)

    print("Train/test difference summary saved to reports/results/train_test_difference_summary.csv")


def generate_eda_report_text():
    report_path = RESULT_DIR / "eda_report_notes.md"

    content = """# EDA Report Notes

## Nội dung đã phân tích

1. Tổng quan số dòng, số cột của train/test.
2. Kiểm tra kiểu dữ liệu và số lượng giá trị duy nhất.
3. Kiểm tra missing values.
4. Kiểm tra duplicate rows.
5. Phân phối nhãn Normal/Attack.
6. Phân phối các loại tấn công trong `attack_cat`.
7. Phân tích các thuộc tính phân loại như `proto`, `service`, `state`.
8. Phân tích các thuộc tính số.
9. So sánh giá trị trung bình của các đặc trưng giữa Normal và Attack.
10. Phân tích tương quan giữa các đặc trưng số và nhãn `label`.
11. So sánh khác biệt cơ bản giữa train và test.

## Nhận xét dùng cho báo cáo

- Dataset có thể dùng cho bài toán phân loại nhị phân với `label`: 0 là Normal, 1 là Attack.
- Cột `attack_cat` có thể dùng cho phân tích phân phối loại tấn công hoặc mở rộng sang bài toán phân loại đa lớp.
- Các cột dạng chuỗi như `proto`, `service`, `state` cần được mã hóa trước khi đưa vào mô hình học máy.
- Các cột số có thang đo khác nhau nên cần cân nhắc chuẩn hóa khi dùng các mô hình như Logistic Regression.
- Với các mô hình cây như Decision Tree, Random Forest, XGBoost hoặc LightGBM, việc chuẩn hóa không bắt buộc nhưng vẫn cần xử lý dữ liệu phân loại.
- Nếu phân phối nhãn mất cân bằng, cần ưu tiên Recall và F1-score của lớp Attack thay vì chỉ nhìn Accuracy.
"""

    report_path.write_text(content, encoding="utf-8")
    print(f"EDA report notes saved to: {report_path}")


def main():
    train_df, test_df = load_data()

    basic_overview(train_df, test_df)
    missing_value_analysis(train_df, test_df)
    duplicate_analysis(train_df, test_df)
    label_distribution(train_df, test_df)
    attack_category_distribution(train_df, test_df)
    categorical_feature_analysis(train_df)
    numerical_feature_analysis(train_df)
    feature_by_label_analysis(train_df)
    correlation_analysis(train_df)
    train_test_difference_analysis(train_df, test_df)
    generate_eda_report_text()

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Result tables saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()