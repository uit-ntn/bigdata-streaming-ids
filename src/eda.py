from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

FIGURE_DIR = REPORT_DIR / "figures" / "eda"
RESULT_DIR = REPORT_DIR / "results" / "eda"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

FEATURE_PATH_OPTIONS = [
    RAW_DIR / "NUSW_NB15_features.csv",
    RAW_DIR / "NUSW-NB15_features.csv",
    RAW_DIR / "UNSW_NB15_features.csv",
    RAW_DIR / "UNSW-NB15_features.csv",
]

TARGET_COL = "label"
ATTACK_COL = "attack_cat"


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_figure(filename: str):
    plt.tight_layout()
    save_path = FIGURE_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


def save_json(data, filename: str):
    save_path = RESULT_DIR / filename
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)
    print(f"[OK] Saved result: {save_path}")


def get_feature_path():
    for path in FEATURE_PATH_OPTIONS:
        if path.exists():
            return path
    return None


def safe_log1p(series: pd.Series):
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series >= 0]
    return np.log1p(series)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    print_section("1. LOAD DATA")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing testing file: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_path = get_feature_path()
    features_df = None

    if feature_path is not None:
        try:
            features_df = pd.read_csv(feature_path, encoding="latin1")
            print(f"[OK] Loaded feature description: {feature_path.name}")
        except Exception as e:
            print(f"[WARNING] Could not load feature description file: {e}")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    return train_df, test_df, features_df


# ============================================================
# 4. BASIC OVERVIEW
# ============================================================

def basic_overview(train_df, test_df, features_df):
    print_section("2. BASIC OVERVIEW")

    overview_df = pd.DataFrame({
        "dataset": ["train", "test"],
        "rows": [train_df.shape[0], test_df.shape[0]],
        "columns": [train_df.shape[1], test_df.shape[1]],
    })

    print(overview_df)
    overview_df.to_csv(RESULT_DIR / "dataset_overview.csv", index=False)

    column_summary = pd.DataFrame({
        "column": train_df.columns,
        "dtype": train_df.dtypes.astype(str).values,
        "non_null_count": train_df.notnull().sum().values,
        "null_count": train_df.isnull().sum().values,
        "unique_count": train_df.nunique().values,
    })

    column_summary.to_csv(RESULT_DIR / "column_summary.csv", index=False)

    if features_df is not None:
        features_df.to_csv(RESULT_DIR / "feature_description_preview.csv", index=False)

    print("[OK] Saved dataset overview and column summary.")


def plot_dataset_size(train_df, test_df):
    labels = ["Train", "Test"]
    values = [len(train_df), len(test_df)]

    plt.figure(figsize=(7, 4))
    plt.barh(labels, values)

    for i, value in enumerate(values):
        plt.text(value, i, f" {value:,}", va="center")

    plt.title("Dataset Size: Train vs Test")
    plt.xlabel("Number of Records")

    save_figure("01_dataset_size_horizontal_bar.png")


# ============================================================
# 5. MISSING VALUES AND DUPLICATES
# ============================================================

def missing_value_analysis(train_df, test_df):
    print_section("3. MISSING VALUE ANALYSIS")

    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()

    missing_df = pd.DataFrame({
        "column": train_df.columns,
        "train_missing": train_missing.values,
        "train_missing_ratio_percent": train_missing.values / len(train_df) * 100,
        "test_missing": test_missing.reindex(train_df.columns, fill_value=0).values,
        "test_missing_ratio_percent": (
            test_missing.reindex(train_df.columns, fill_value=0).values / len(test_df) * 100
        ),
    })

    missing_nonzero_df = missing_df[
        (missing_df["train_missing"] > 0) | (missing_df["test_missing"] > 0)
    ].sort_values("train_missing", ascending=False)

    missing_df.to_csv(RESULT_DIR / "missing_values_all_columns.csv", index=False)
    missing_nonzero_df.to_csv(RESULT_DIR / "missing_values_nonzero.csv", index=False)

    if missing_nonzero_df.empty:
        print("No missing values found.")
    else:
        print(missing_nonzero_df.head(20))


def plot_missing_values(train_df, test_df):
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()

    missing_df = pd.DataFrame({
        "column": train_df.columns,
        "train_missing": train_missing.values,
        "test_missing": test_missing.reindex(train_df.columns, fill_value=0).values,
    })

    missing_df = missing_df[
        (missing_df["train_missing"] > 0) | (missing_df["test_missing"] > 0)
    ].sort_values("train_missing", ascending=True)

    if missing_df.empty:
        plt.figure(figsize=(8, 3))
        plt.text(
            0.5,
            0.5,
            "No missing values found",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.axis("off")
        plt.title("Missing Value Analysis")
        save_figure("02_missing_values_status.png")
        return

    top_df = missing_df.tail(20)

    plt.figure(figsize=(9, 6))
    y = np.arange(len(top_df))
    plt.barh(y - 0.2, top_df["train_missing"], height=0.4, label="Train")
    plt.barh(y + 0.2, top_df["test_missing"], height=0.4, label="Test")
    plt.yticks(y, top_df["column"])
    plt.xlabel("Missing Count")
    plt.title("Top Missing Values: Train vs Test")
    plt.legend()

    save_figure("02_missing_values_grouped_horizontal_bar.png")


def duplicate_analysis(train_df, test_df):
    print_section("4. DUPLICATE ANALYSIS")

    duplicate_df = pd.DataFrame({
        "dataset": ["train", "test"],
        "duplicate_rows": [train_df.duplicated().sum(), test_df.duplicated().sum()],
        "total_rows": [len(train_df), len(test_df)],
    })

    duplicate_df["duplicate_ratio_percent"] = (
        duplicate_df["duplicate_rows"] / duplicate_df["total_rows"] * 100
    )

    print(duplicate_df)
    duplicate_df.to_csv(RESULT_DIR / "duplicate_summary.csv", index=False)


# ============================================================
# 6. LABEL DISTRIBUTION
# ============================================================

def label_distribution(train_df, test_df):
    print_section("5. LABEL DISTRIBUTION")

    train_counts = train_df[TARGET_COL].value_counts().sort_index()
    test_counts = test_df[TARGET_COL].value_counts().sort_index()

    labels = sorted(set(train_counts.index).union(set(test_counts.index)))

    label_df = pd.DataFrame({
        "label": labels,
        "label_name": ["Normal" if label == 0 else "Attack" for label in labels],
        "train_count": [train_counts.get(label, 0) for label in labels],
        "test_count": [test_counts.get(label, 0) for label in labels],
    })

    label_df["train_ratio_percent"] = label_df["train_count"] / len(train_df) * 100
    label_df["test_ratio_percent"] = label_df["test_count"] / len(test_df) * 100

    print(label_df)
    label_df.to_csv(RESULT_DIR / "label_distribution.csv", index=False)

    return label_df


def plot_label_distribution_pie(label_df):
    plt.figure(figsize=(7, 6))
    plt.pie(
        label_df["train_count"],
        labels=label_df["label_name"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Label Distribution in Training Set")

    save_figure("03_label_distribution_pie.png")


def plot_train_test_label_stacked_bar(label_df):
    train_normal = float(label_df.loc[label_df["label"] == 0, "train_ratio_percent"].sum())
    train_attack = float(label_df.loc[label_df["label"] == 1, "train_ratio_percent"].sum())
    test_normal = float(label_df.loc[label_df["label"] == 0, "test_ratio_percent"].sum())
    test_attack = float(label_df.loc[label_df["label"] == 1, "test_ratio_percent"].sum())

    x = np.arange(2)

    plt.figure(figsize=(7, 5))
    plt.bar(x, [train_normal, test_normal], label="Normal")
    plt.bar(x, [train_attack, test_attack], bottom=[train_normal, test_normal], label="Attack")

    plt.xticks(x, ["Train", "Test"])
    plt.ylabel("Ratio (%)")
    plt.title("Train/Test Label Ratio Comparison")
    plt.legend()

    save_figure("04_train_test_label_stacked_bar.png")


# ============================================================
# 7. ATTACK CATEGORY DISTRIBUTION
# ============================================================

def attack_category_distribution(train_df, test_df):
    print_section("6. ATTACK CATEGORY DISTRIBUTION")

    if ATTACK_COL not in train_df.columns:
        print(f"Column '{ATTACK_COL}' not found.")
        return None

    train_counts = train_df[ATTACK_COL].value_counts()
    test_counts = test_df[ATTACK_COL].value_counts()

    all_categories = sorted(set(train_counts.index).union(set(test_counts.index)))

    attack_df = pd.DataFrame({
        "attack_cat": all_categories,
        "train_count": [train_counts.get(cat, 0) for cat in all_categories],
        "test_count": [test_counts.get(cat, 0) for cat in all_categories],
    })

    attack_df["train_ratio_percent"] = attack_df["train_count"] / len(train_df) * 100
    attack_df["test_ratio_percent"] = attack_df["test_count"] / len(test_df) * 100

    attack_df = attack_df.sort_values("train_count", ascending=False)

    print(attack_df)
    attack_df.to_csv(RESULT_DIR / "attack_category_distribution.csv", index=False)

    return attack_df


def plot_attack_category_horizontal_bar(attack_df):
    if attack_df is None:
        return

    plot_df = attack_df.sort_values("train_count", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["attack_cat"], plot_df["train_count"])

    for i, value in enumerate(plot_df["train_count"]):
        plt.text(value, i, f" {value:,}", va="center", fontsize=8)

    plt.title("Attack Category Distribution in Training Set")
    plt.xlabel("Number of Records")
    plt.ylabel("Attack Category")

    save_figure("05_attack_category_horizontal_bar.png")


def plot_attack_category_pie_top(attack_df):
    if attack_df is None:
        return

    top_n = 6
    top_df = attack_df.head(top_n).copy()
    other_count = attack_df.iloc[top_n:]["train_count"].sum()

    if other_count > 0:
        other_df = pd.DataFrame([{
            "attack_cat": "Others",
            "train_count": other_count,
        }])
        top_df = pd.concat([top_df, other_df], ignore_index=True)

    plt.figure(figsize=(8, 7))
    plt.pie(
        top_df["train_count"],
        labels=top_df["attack_cat"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Top Attack Categories Ratio in Training Set")

    save_figure("06_attack_category_top_pie.png")


def plot_train_test_attack_ratio_grouped(attack_df):
    if attack_df is None:
        return

    plot_df = attack_df.sort_values("train_count", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df["train_ratio_percent"], width, label="Train")
    plt.bar(x + width / 2, plot_df["test_ratio_percent"], width, label="Test")

    plt.xticks(x, plot_df["attack_cat"], rotation=45, ha="right")
    plt.ylabel("Ratio (%)")
    plt.title("Attack Category Ratio: Train vs Test")
    plt.legend()

    save_figure("07_attack_category_train_test_grouped_bar.png")


# ============================================================
# 8. CATEGORICAL FEATURES
# ============================================================

def categorical_feature_analysis(train_df):
    print_section("7. CATEGORICAL FEATURE ANALYSIS")

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    summary = []

    for col in categorical_cols:
        counts = train_df[col].value_counts()

        summary.append({
            "column": col,
            "unique_count": train_df[col].nunique(),
            "top_value": counts.index[0],
            "top_count": counts.iloc[0],
            "top_ratio_percent": counts.iloc[0] / len(train_df) * 100,
        })

        counts.head(20).to_csv(RESULT_DIR / f"categorical_top_values_{col}.csv")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULT_DIR / "categorical_summary.csv", index=False)

    print(summary_df)

    return categorical_cols, summary_df


def plot_categorical_cardinality(summary_df):
    if summary_df.empty:
        return

    plot_df = summary_df.sort_values("unique_count", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["column"], plot_df["unique_count"])

    for i, value in enumerate(plot_df["unique_count"]):
        plt.text(value, i, f" {value}", va="center")

    plt.title("Number of Unique Values in Categorical Features")
    plt.xlabel("Unique Count")
    plt.ylabel("Categorical Feature")

    save_figure("08_categorical_cardinality_horizontal_bar.png")


def plot_top_categorical_values(train_df, columns=None):
    if columns is None:
        columns = ["proto", "service", "state"]

    for col in columns:
        if col not in train_df.columns:
            continue

        counts = train_df[col].value_counts().head(15).sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(counts.index.astype(str), counts.values)

        for i, value in enumerate(counts.values):
            plt.text(value, i, f" {value:,}", va="center", fontsize=8)

        plt.title(f"Top 15 Values of {col}")
        plt.xlabel("Count")
        plt.ylabel(col)

        save_figure(f"09_top_values_{col}_horizontal_bar.png")


# ============================================================
# 9. NUMERICAL FEATURES
# ============================================================

def numerical_feature_analysis(train_df):
    print_section("8. NUMERICAL FEATURE ANALYSIS")

    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    numerical_summary = train_df[numerical_cols].describe().T
    numerical_summary["missing"] = train_df[numerical_cols].isnull().sum()
    numerical_summary["unique_count"] = train_df[numerical_cols].nunique()

    numerical_summary.to_csv(RESULT_DIR / "numerical_summary.csv")

    print(numerical_summary.head(20))

    return numerical_cols, numerical_summary


def plot_numeric_histograms(train_df, numerical_cols):
    selected_cols = [
        "dur",
        "sbytes",
        "dbytes",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "smean",
        "dmean",
    ]

    selected_cols = [col for col in selected_cols if col in numerical_cols]

    for col in selected_cols[:6]:
        values = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        save_figure(f"10_histogram_{col}.png")

        log_values = safe_log1p(train_df[col])

        plt.figure(figsize=(8, 5))
        plt.hist(log_values, bins=50)
        plt.title(f"Log Distribution of {col}")
        plt.xlabel(f"log1p({col})")
        plt.ylabel("Frequency")

        save_figure(f"11_log_histogram_{col}.png")


def plot_numeric_boxplots_by_label(train_df, numerical_cols):
    selected_cols = [
        "dur",
        "sbytes",
        "dbytes",
        "sload",
        "dload",
        "spkts",
        "dpkts",
    ]

    selected_cols = [
        col for col in selected_cols
        if col in numerical_cols and TARGET_COL in train_df.columns
    ]

    for col in selected_cols[:5]:
        normal_values = safe_log1p(train_df.loc[train_df[TARGET_COL] == 0, col])
        attack_values = safe_log1p(train_df.loc[train_df[TARGET_COL] == 1, col])

        plt.figure(figsize=(7, 5))
        plt.boxplot(
            [normal_values, attack_values],
            labels=["Normal", "Attack"],
            showfliers=False,
        )
        plt.title(f"Boxplot of log1p({col}) by Label")
        plt.xlabel("Label")
        plt.ylabel(f"log1p({col})")

        save_figure(f"12_boxplot_log_{col}_by_label.png")


def plot_numeric_scatter_sample(train_df):
    required_cols = ["sbytes", "dbytes", TARGET_COL]

    if not all(col in train_df.columns for col in required_cols):
        return

    sample_df = train_df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()

    sample_size = min(5000, len(sample_df))
    sample_df = sample_df.sample(sample_size, random_state=42)

    normal_df = sample_df[sample_df[TARGET_COL] == 0]
    attack_df = sample_df[sample_df[TARGET_COL] == 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        np.log1p(normal_df["sbytes"]),
        np.log1p(normal_df["dbytes"]),
        alpha=0.35,
        label="Normal",
        s=10,
    )

    plt.scatter(
        np.log1p(attack_df["sbytes"]),
        np.log1p(attack_df["dbytes"]),
        alpha=0.35,
        label="Attack",
        s=10,
    )

    plt.title("Scatter Sample: log1p(sbytes) vs log1p(dbytes)")
    plt.xlabel("log1p(sbytes)")
    plt.ylabel("log1p(dbytes)")
    plt.legend()

    save_figure("13_scatter_sbytes_dbytes_by_label.png")


# ============================================================
# 10. FEATURE BY LABEL
# ============================================================

def feature_by_label_analysis(train_df, numerical_cols):
    print_section("9. FEATURE BY LABEL ANALYSIS")

    ignored_cols = ["id", TARGET_COL]
    numeric_features = [col for col in numerical_cols if col not in ignored_cols]

    group_summary = train_df.groupby(TARGET_COL)[numeric_features].mean().T

    rename_map = {}

    if 0 in group_summary.columns:
        rename_map[0] = "normal_mean"

    if 1 in group_summary.columns:
        rename_map[1] = "attack_mean"

    group_summary = group_summary.rename(columns=rename_map)

    if "normal_mean" in group_summary.columns and "attack_mean" in group_summary.columns:
        group_summary["difference"] = group_summary["attack_mean"] - group_summary["normal_mean"]
        group_summary["abs_difference"] = group_summary["difference"].abs()
        group_summary = group_summary.sort_values("abs_difference", ascending=False)

    group_summary.to_csv(RESULT_DIR / "feature_mean_by_label.csv")

    print(group_summary.head(15))

    return group_summary


def plot_top_mean_difference(group_summary):
    if group_summary is None or group_summary.empty or "abs_difference" not in group_summary.columns:
        return

    plot_df = group_summary.head(15).sort_values("abs_difference", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df.index, plot_df["abs_difference"])

    plt.title("Top Numerical Features by Mean Difference Between Normal and Attack")
    plt.xlabel("Absolute Mean Difference")
    plt.ylabel("Feature")

    save_figure("14_top_feature_mean_difference_horizontal_bar.png")


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(train_df):
    print_section("10. CORRELATION ANALYSIS")

    numerical_df = train_df.select_dtypes(include=[np.number]).copy()

    if "id" in numerical_df.columns:
        numerical_df = numerical_df.drop(columns=["id"])

    corr = numerical_df.corr()
    corr.to_csv(RESULT_DIR / "correlation_matrix.csv")

    label_corr = None

    if TARGET_COL in corr.columns:
        label_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
        label_corr.to_csv(RESULT_DIR / "label_correlation.csv")
        print(label_corr.head(20))

    return corr, label_corr


def plot_label_correlation(label_corr):
    if label_corr is None or label_corr.empty:
        return

    plot_df = label_corr.head(20).sort_values()

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df.index, plot_df.values)

    plt.title("Top 20 Features Correlated with Label")
    plt.xlabel("Correlation with Label")
    plt.ylabel("Feature")

    save_figure("15_top_label_correlation_horizontal_bar.png")


def plot_correlation_heatmap(train_df):
    numerical_df = train_df.select_dtypes(include=[np.number]).copy()

    if "id" in numerical_df.columns:
        numerical_df = numerical_df.drop(columns=["id"])

    if TARGET_COL not in numerical_df.columns:
        return

    corr = numerical_df.corr()
    selected_cols = corr[TARGET_COL].abs().sort_values(ascending=False).head(15).index.tolist()

    corr_subset = numerical_df[selected_cols].corr()

    plt.figure(figsize=(11, 9))
    plt.imshow(corr_subset, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(selected_cols)), selected_cols, rotation=90)
    plt.yticks(range(len(selected_cols)), selected_cols)
    plt.title("Correlation Heatmap of Top Label-related Features")

    save_figure("16_correlation_heatmap_top_features.png")


# ============================================================
# 12. TRAIN / TEST DIFFERENCE
# ============================================================

def train_test_difference_analysis(train_df, test_df):
    print_section("11. TRAIN/TEST DIFFERENCE ANALYSIS")

    common_cols = [col for col in train_df.columns if col in test_df.columns]
    rows = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            rows.append({
                "column": col,
                "type": "numerical",
                "train_mean": train_df[col].mean(),
                "test_mean": test_df[col].mean(),
                "mean_diff": test_df[col].mean() - train_df[col].mean(),
                "train_std": train_df[col].std(),
                "test_std": test_df[col].std(),
            })
        else:
            rows.append({
                "column": col,
                "type": "categorical",
                "train_unique": train_df[col].nunique(),
                "test_unique": test_df[col].nunique(),
            })

    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(RESULT_DIR / "train_test_difference_summary.csv", index=False)

    print("[OK] Saved train/test difference summary.")

    return diff_df


# ============================================================
# 13. GENERATE REPORT NOTES
# ============================================================

def generate_eda_report_notes(train_df, test_df):
    print_section("12. GENERATE EDA REPORT NOTES")

    label_counts = train_df[TARGET_COL].value_counts().sort_index()

    normal_count = int(label_counts.get(0, 0))
    attack_count = int(label_counts.get(1, 0))

    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    lines = [
        "# EDA Report Notes",
        "",
        "## 1. Tổng quan dữ liệu",
        "",
        f"Tập train có {train_df.shape[0]:,} dòng và {train_df.shape[1]} cột.",
        f"Tập test có {test_df.shape[0]:,} dòng và {test_df.shape[1]} cột.",
        "",
        "## 2. Nhãn phân loại",
        "",
        f"Bài toán hiện tại sử dụng cột `{TARGET_COL}` cho phân loại nhị phân:",
        "",
        "- `0`: Normal",
        "- `1`: Attack",
        "",
        "Phân phối nhãn trong tập train:",
        "",
        f"- Normal: {normal_count:,}",
        f"- Attack: {attack_count:,}",
        "",
        "## 3. Thuộc tính phân loại và thuộc tính số",
        "",
        f"Số cột phân loại: {len(categorical_cols)}",
        f"Các cột phân loại: {categorical_cols}",
        "",
        f"Số cột số: {len(numerical_cols)}",
        "",
        "Các cột phân loại như `proto`, `service`, `state` cần được mã hóa trước khi train model.",
        "Các cột số có thang đo khác nhau nên cần chuẩn hóa nếu sử dụng các mô hình nhạy với scale như Logistic Regression.",
        "",
        "## 4. Biểu đồ đã xuất",
        "",
        "Các biểu đồ EDA được lưu trong:",
        "",
        "reports/figures/eda/",
        "",
        "Các bảng kết quả EDA được lưu trong:",
        "",
        "reports/results/eda/",
        "",
        "## 5. Nhận xét dùng cho báo cáo",
        "",
        "- Dataset phù hợp cho bài toán phát hiện xâm nhập mạng với nhãn Normal/Attack.",
        "- Cột `attack_cat` có thể dùng để phân tích loại tấn công hoặc mở rộng sang bài toán phân loại đa lớp.",
        "- Phân phối giữa các loại tấn công có thể không đồng đều, do đó khi đánh giá mô hình nên ưu tiên Recall và F1-score của lớp Attack.",
        "- Các cột phân loại cần được One-Hot Encoding hoặc encoding phù hợp trước khi huấn luyện.",
        "- Một số đặc trưng số có phân phối lệch mạnh, vì vậy việc trực quan hóa bằng log scale giúp dễ quan sát hơn.",
    ]

    content = "\n".join(lines)

    report_path = RESULT_DIR / "eda_report_notes.md"
    report_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved report notes: {report_path}")


# ============================================================
# 14. MAIN
# ============================================================

def main():
    train_df, test_df, features_df = load_data()

    basic_overview(train_df, test_df, features_df)
    plot_dataset_size(train_df, test_df)

    missing_value_analysis(train_df, test_df)
    plot_missing_values(train_df, test_df)

    duplicate_analysis(train_df, test_df)

    label_df = label_distribution(train_df, test_df)
    plot_label_distribution_pie(label_df)
    plot_train_test_label_stacked_bar(label_df)

    attack_df = attack_category_distribution(train_df, test_df)
    plot_attack_category_horizontal_bar(attack_df)
    plot_attack_category_pie_top(attack_df)
    plot_train_test_attack_ratio_grouped(attack_df)

    categorical_cols, categorical_summary = categorical_feature_analysis(train_df)
    plot_categorical_cardinality(categorical_summary)
    plot_top_categorical_values(train_df)

    numerical_cols, numerical_summary = numerical_feature_analysis(train_df)
    plot_numeric_histograms(train_df, numerical_cols)
    plot_numeric_boxplots_by_label(train_df, numerical_cols)
    plot_numeric_scatter_sample(train_df)

    group_summary = feature_by_label_analysis(train_df, numerical_cols)
    plot_top_mean_difference(group_summary)

    corr, label_corr = correlation_analysis(train_df)
    plot_label_correlation(label_corr)
    plot_correlation_heatmap(train_df)

    train_test_difference_analysis(train_df, test_df)

    generate_eda_report_notes(train_df, test_df)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Result tables saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()