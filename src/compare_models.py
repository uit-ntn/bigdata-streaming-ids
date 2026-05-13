from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

MODEL_RESULT_BASE_DIR = REPORT_DIR / "results" / "models"
MODEL_FIGURE_BASE_DIR = REPORT_DIR / "figures" / "models"

COMPARISON_RESULT_DIR = REPORT_DIR / "results" / "model_comparison"
COMPARISON_FIGURE_DIR = REPORT_DIR / "figures" / "model_comparison"

COMPARISON_RESULT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "type": "Linear baseline",
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "type": "Non-linear baseline",
    },
    "deep_mlp": {
        "display_name": "Deep MLP",
        "type": "Deep Learning",
    },
}


METRIC_COLUMNS = [
    "accuracy",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "pr_auc",
]


KEY_METRICS_FOR_REPORT = [
    "accuracy",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "roc_auc",
    "pr_auc",
]


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
    save_path = COMPARISON_FIGURE_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved figure: {save_path}")


def read_json_if_exists(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_model_outputs():
    print_section("1. CHECK REQUIRED MODEL OUTPUTS")

    missing_items = []

    for model_name in MODELS:
        model_dir = MODEL_RESULT_BASE_DIR / model_name
        metrics_path = model_dir / "metrics_summary.csv"

        if not model_dir.exists():
            missing_items.append(str(model_dir))

        if not metrics_path.exists():
            missing_items.append(str(metrics_path))

    if missing_items:
        print("Missing required files/folders:")
        for item in missing_items:
            print(f"- {item}")

        raise FileNotFoundError(
            "Some model results are missing. "
            "Please run all model training scripts before comparing models."
        )

    print("[OK] All required model result folders found.")


# ============================================================
# 3. LOAD MODEL RESULTS
# ============================================================

def load_model_metrics():
    print_section("2. LOAD MODEL METRICS")

    rows = []

    for model_name, model_info in MODELS.items():
        model_result_dir = MODEL_RESULT_BASE_DIR / model_name
        metrics_path = model_result_dir / "metrics_summary.csv"
        config_path = model_result_dir / "model_config.json"
        training_info_path = model_result_dir / "training_info.json"
        prediction_info_path = model_result_dir / "prediction_time_info.json"

        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            raise ValueError(f"metrics_summary.csv is empty: {metrics_path}")

        metrics = metrics_df.iloc[0].to_dict()

        config = read_json_if_exists(config_path)
        training_info = read_json_if_exists(training_info_path)
        prediction_info = read_json_if_exists(prediction_info_path)

        row = {
            "model": model_name,
            "display_name": model_info["display_name"],
            "model_type": model_info["type"],
        }

        for col in metrics_df.columns:
            row[col] = metrics[col]

        if config is not None:
            row["algorithm"] = config.get("algorithm", model_info["display_name"])
        else:
            row["algorithm"] = model_info["display_name"]

        if training_info is not None:
            row["epochs_completed"] = training_info.get("epochs_completed", np.nan)
            row["tree_depth"] = training_info.get("tree_depth", row.get("tree_depth", np.nan))
            row["num_leaves"] = training_info.get("num_leaves", row.get("num_leaves", np.nan))

        if prediction_info is not None:
            row["prediction_time_per_sample_ms"] = prediction_info.get(
                "prediction_time_per_sample_ms",
                np.nan,
            )

        rows.append(row)

        print(f"[OK] Loaded metrics for {model_info['display_name']}")

    comparison_df = pd.DataFrame(rows)

    comparison_df.to_csv(
        COMPARISON_RESULT_DIR / "model_metrics_comparison_raw.csv",
        index=False,
    )

    return comparison_df


# ============================================================
# 4. COMPUTE RANKING
# ============================================================

def compute_model_ranking(comparison_df: pd.DataFrame):
    print_section("3. COMPUTE MODEL RANKING")

    df = comparison_df.copy()

    score_weights = {
        "recall_attack": 0.35,
        "f1_attack": 0.30,
        "precision_attack": 0.15,
        "roc_auc": 0.10,
        "pr_auc": 0.10,
    }

    for metric in score_weights:
        if metric not in df.columns:
            raise ValueError(f"Missing metric for ranking: {metric}")

    df["weighted_score"] = 0.0

    for metric, weight in score_weights.items():
        df["weighted_score"] += df[metric].astype(float) * weight

    df["rank_by_weighted_score"] = df["weighted_score"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    df["rank_by_recall_attack"] = df["recall_attack"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    df["rank_by_f1_attack"] = df["f1_attack"].rank(
        method="dense",
        ascending=False,
    ).astype(int)

    df = df.sort_values(
        by=["weighted_score", "recall_attack", "f1_attack"],
        ascending=False,
    )

    selected_cols = [
        "rank_by_weighted_score",
        "display_name",
        "model_type",
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "pr_auc",
        "train_time_seconds",
        "prediction_time_seconds",
        "prediction_time_per_sample_ms",
        "weighted_score",
    ]

    selected_cols = [col for col in selected_cols if col in df.columns]

    ranking_df = df[selected_cols]

    ranking_df.to_csv(
        COMPARISON_RESULT_DIR / "model_ranking.csv",
        index=False,
    )

    score_config = {
        "score_formula": "0.35*recall_attack + 0.30*f1_attack + 0.15*precision_attack + 0.10*roc_auc + 0.10*pr_auc",
        "weights": score_weights,
        "reason": (
            "IDS evaluation prioritizes Recall and F1-score for the Attack class "
            "because missing attacks is more risky than only maximizing overall accuracy."
        ),
    }

    save_json(score_config, COMPARISON_RESULT_DIR / "ranking_score_config.json")

    print(ranking_df)

    return df, ranking_df


# ============================================================
# 5. PLOT METRIC COMPARISON
# ============================================================

def plot_grouped_metric_comparison(ranking_df):
    plot_metrics = [
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "pr_auc",
    ]

    plot_df = ranking_df.set_index("display_name")[plot_metrics]

    x = np.arange(len(plot_df.index))
    width = 0.12

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(plot_metrics):
        plt.bar(
            x + (i - len(plot_metrics) / 2) * width + width / 2,
            plot_df[metric],
            width,
            label=metric,
        )

    plt.xticks(x, plot_df.index, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.title("Model Performance Comparison")
    plt.legend(loc="lower right")

    save_figure("01_grouped_metric_comparison.png")


def plot_attack_priority_metrics(ranking_df):
    plot_df = ranking_df.set_index("display_name")[[
        "precision_attack",
        "recall_attack",
        "f1_attack",
    ]]

    x = np.arange(len(plot_df.index))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, plot_df["precision_attack"], width, label="Precision Attack")
    plt.bar(x, plot_df["recall_attack"], width, label="Recall Attack")
    plt.bar(x + width, plot_df["f1_attack"], width, label="F1 Attack")

    plt.xticks(x, plot_df.index, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value")
    plt.title("Attack Class Metrics Comparison")
    plt.legend()

    save_figure("02_attack_priority_metrics.png")


def plot_weighted_score_lollipop(ranking_df):
    plot_df = ranking_df.sort_values("weighted_score", ascending=True)

    y_pos = np.arange(len(plot_df))

    plt.figure(figsize=(9, 5))
    plt.hlines(
        y=y_pos,
        xmin=0,
        xmax=plot_df["weighted_score"],
        linewidth=2,
    )
    plt.plot(plot_df["weighted_score"], y_pos, "o")

    plt.yticks(y_pos, plot_df["display_name"])
    plt.xlabel("Weighted Score")
    plt.title("Overall Model Ranking Score")

    for i, score in enumerate(plot_df["weighted_score"]):
        plt.text(score, i, f" {score:.4f}", va="center")

    save_figure("03_weighted_score_lollipop.png")


def plot_roc_auc_pr_auc_comparison(ranking_df):
    plot_df = ranking_df.set_index("display_name")[["roc_auc", "pr_auc"]]

    x = np.arange(len(plot_df.index))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, plot_df["roc_auc"], width, label="ROC-AUC")
    plt.bar(x + width / 2, plot_df["pr_auc"], width, label="PR-AUC")

    plt.xticks(x, plot_df.index, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("AUC Value")
    plt.title("ROC-AUC and PR-AUC Comparison")
    plt.legend()

    save_figure("04_roc_auc_pr_auc_comparison.png")


# ============================================================
# 6. PLOT TIME COMPARISON
# ============================================================

def plot_training_time_comparison(ranking_df):
    if "train_time_seconds" not in ranking_df.columns:
        return

    plot_df = ranking_df.sort_values("train_time_seconds", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["display_name"], plot_df["train_time_seconds"])

    for i, value in enumerate(plot_df["train_time_seconds"]):
        plt.text(value, i, f" {value:.2f}s", va="center")

    plt.xlabel("Training Time (seconds)")
    plt.title("Training Time Comparison")

    save_figure("05_training_time_comparison.png")


def plot_prediction_time_comparison(ranking_df):
    if "prediction_time_seconds" not in ranking_df.columns:
        return

    plot_df = ranking_df.sort_values("prediction_time_seconds", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["display_name"], plot_df["prediction_time_seconds"])

    for i, value in enumerate(plot_df["prediction_time_seconds"]):
        plt.text(value, i, f" {value:.4f}s", va="center")

    plt.xlabel("Prediction Time (seconds)")
    plt.title("Prediction Time Comparison on Test Set")

    save_figure("06_prediction_time_comparison.png")


def plot_prediction_time_per_sample(ranking_df):
    if "prediction_time_per_sample_ms" not in ranking_df.columns:
        return

    plot_df = ranking_df.dropna(subset=["prediction_time_per_sample_ms"]).sort_values(
        "prediction_time_per_sample_ms",
        ascending=True,
    )

    if plot_df.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["display_name"], plot_df["prediction_time_per_sample_ms"])

    for i, value in enumerate(plot_df["prediction_time_per_sample_ms"]):
        plt.text(value, i, f" {value:.6f} ms", va="center")

    plt.xlabel("Prediction Time per Sample (ms)")
    plt.title("Prediction Time per Sample Comparison")

    save_figure("07_prediction_time_per_sample.png")


# ============================================================
# 7. CONFUSION MATRIX COMPONENT COMPARISON
# ============================================================

def plot_confusion_components(ranking_df):
    required_cols = ["tn", "fp", "fn", "tp"]

    if not all(col in ranking_df.columns for col in required_cols):
        return

    plot_df = ranking_df.set_index("display_name")[required_cols]

    x = np.arange(len(plot_df.index))
    width = 0.2

    plt.figure(figsize=(11, 6))
    plt.bar(x - 1.5 * width, plot_df["tn"], width, label="TN")
    plt.bar(x - 0.5 * width, plot_df["fp"], width, label="FP")
    plt.bar(x + 0.5 * width, plot_df["fn"], width, label="FN")
    plt.bar(x + 1.5 * width, plot_df["tp"], width, label="TP")

    plt.xticks(x, plot_df.index, rotation=15, ha="right")
    plt.ylabel("Number of Samples")
    plt.title("Confusion Matrix Components by Model")
    plt.legend()

    save_figure("08_confusion_components_grouped_bar.png")


def plot_false_negative_comparison(ranking_df):
    if "fn" not in ranking_df.columns:
        return

    plot_df = ranking_df.sort_values("fn", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["display_name"], plot_df["fn"])

    for i, value in enumerate(plot_df["fn"]):
        plt.text(value, i, f" {int(value)}", va="center")

    plt.xlabel("False Negative Count")
    plt.title("False Negative Comparison: Missed Attacks")

    save_figure("09_false_negative_comparison.png")


def plot_false_positive_comparison(ranking_df):
    if "fp" not in ranking_df.columns:
        return

    plot_df = ranking_df.sort_values("fp", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["display_name"], plot_df["fp"])

    for i, value in enumerate(plot_df["fp"]):
        plt.text(value, i, f" {int(value)}", va="center")

    plt.xlabel("False Positive Count")
    plt.title("False Positive Comparison: False Alarms")

    save_figure("10_false_positive_comparison.png")


# ============================================================
# 8. RADAR CHART
# ============================================================

def plot_radar_chart(ranking_df):
    metrics_for_radar = [
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "pr_auc",
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    num_vars = len(metrics_for_radar)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in ranking_df.iterrows():
        values = [row[metric] for metric in metrics_for_radar]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["display_name"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison Radar Chart", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_path = COMPARISON_FIGURE_DIR / "11_model_comparison_radar_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {save_path}")


# ============================================================
# 9. COMBINED ROC AND PR CURVES
# ============================================================

def load_curve_points(model_name, curve_file):
    path = MODEL_RESULT_BASE_DIR / model_name / curve_file

    if not path.exists():
        return None

    return pd.read_csv(path)


def plot_combined_roc_curves():
    plt.figure(figsize=(8, 6))

    has_curve = False

    for model_name, model_info in MODELS.items():
        curve_df = load_curve_points(model_name, "roc_curve_points.csv")

        if curve_df is None:
            continue

        plt.plot(
            curve_df["false_positive_rate"],
            curve_df["true_positive_rate"],
            label=model_info["display_name"],
        )

        has_curve = True

    if not has_curve:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    plt.title("Combined ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()

    save_figure("12_combined_roc_curves.png")


def plot_combined_precision_recall_curves():
    plt.figure(figsize=(8, 6))

    has_curve = False

    for model_name, model_info in MODELS.items():
        curve_df = load_curve_points(model_name, "precision_recall_curve_points.csv")

        if curve_df is None:
            continue

        plt.plot(
            curve_df["recall"],
            curve_df["precision"],
            label=model_info["display_name"],
        )

        has_curve = True

    if not has_curve:
        plt.close()
        return

    plt.title("Combined Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    save_figure("13_combined_precision_recall_curves.png")


# ============================================================
# 10. SUMMARY TABLE IMAGE
# ============================================================

def plot_summary_table(ranking_df):
    table_cols = [
        "display_name",
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "pr_auc",
        "weighted_score",
    ]

    table_df = ranking_df[table_cols].copy()

    rename_map = {
        "display_name": "Model",
        "accuracy": "Accuracy",
        "precision_attack": "Precision",
        "recall_attack": "Recall",
        "f1_attack": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "weighted_score": "Weighted Score",
    }

    table_df = table_df.rename(columns=rename_map)

    for col in table_df.columns:
        if col != "Model":
            table_df[col] = table_df[col].astype(float).map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    plt.title("Model Comparison Summary Table")

    save_path = COMPARISON_FIGURE_DIR / "14_model_comparison_summary_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {save_path}")


# ============================================================
# 11. GENERATE REPORT NOTES
# ============================================================

def generate_report_notes(ranking_df):
    print_section("4. GENERATE MODEL COMPARISON REPORT NOTES")

    best_overall = ranking_df.iloc[0]

    best_recall = ranking_df.sort_values(
        "recall_attack",
        ascending=False,
    ).iloc[0]

    best_f1 = ranking_df.sort_values(
        "f1_attack",
        ascending=False,
    ).iloc[0]

    fastest_prediction = None

    if "prediction_time_seconds" in ranking_df.columns:
        fastest_prediction = ranking_df.sort_values(
            "prediction_time_seconds",
            ascending=True,
        ).iloc[0]

    lines = [
        "# Model Comparison Report Notes",
        "",
        "## 1. Mục tiêu so sánh",
        "",
        "Sau khi huấn luyện ba mô hình, kết quả được tổng hợp để so sánh khả năng phát hiện xâm nhập mạng.",
        "Ba mô hình được so sánh gồm Logistic Regression, Decision Tree và Deep MLP.",
        "",
        "## 2. Các chỉ số đánh giá",
        "",
        "Các chỉ số được sử dụng gồm:",
        "",
        "- Accuracy",
        "- Precision cho lớp Attack",
        "- Recall cho lớp Attack",
        "- F1-score cho lớp Attack",
        "- ROC-AUC",
        "- PR-AUC",
        "- Thời gian huấn luyện",
        "- Thời gian dự đoán",
        "- Confusion matrix components: TN, FP, FN, TP",
        "",
        "Trong bài toán IDS, Recall và F1-score của lớp Attack được ưu tiên hơn Accuracy vì bỏ sót tấn công có thể gây rủi ro bảo mật.",
        "",
        "## 3. Công thức điểm tổng hợp",
        "",
        "Điểm tổng hợp được tính theo công thức:",
        "",
        "Weighted Score = 0.35 * Recall_Attack + 0.30 * F1_Attack + 0.15 * Precision_Attack + 0.10 * ROC_AUC + 0.10 * PR_AUC",
        "",
        "Công thức này ưu tiên Recall và F1-score của lớp Attack để phù hợp với bài toán phát hiện xâm nhập mạng.",
        "",
        "## 4. Kết quả tốt nhất",
        "",
        f"Mô hình có Weighted Score cao nhất: {best_overall['display_name']} ({best_overall['weighted_score']:.4f})",
        f"Mô hình có Recall Attack cao nhất: {best_recall['display_name']} ({best_recall['recall_attack']:.4f})",
        f"Mô hình có F1 Attack cao nhất: {best_f1['display_name']} ({best_f1['f1_attack']:.4f})",
    ]

    if fastest_prediction is not None:
        lines.extend([
            f"Mô hình có thời gian dự đoán thấp nhất: {fastest_prediction['display_name']} ({fastest_prediction['prediction_time_seconds']:.6f} giây)",
        ])

    lines.extend([
        "",
        "## 5. Bảng xếp hạng mô hình",
        "",
    ])

    for _, row in ranking_df.iterrows():
        lines.extend([
            f"### {int(row['rank_by_weighted_score'])}. {row['display_name']}",
            "",
            f"- Accuracy: {row['accuracy']:.4f}",
            f"- Precision Attack: {row['precision_attack']:.4f}",
            f"- Recall Attack: {row['recall_attack']:.4f}",
            f"- F1 Attack: {row['f1_attack']:.4f}",
            f"- ROC-AUC: {row['roc_auc']:.4f}",
            f"- PR-AUC: {row['pr_auc']:.4f}",
            f"- Weighted Score: {row['weighted_score']:.4f}",
            "",
        ])

    lines.extend([
        "## 6. Biểu đồ đã xuất",
        "",
        "Các biểu đồ so sánh được lưu tại:",
        "",
        "reports/figures/model_comparison/",
        "",
        "Bao gồm:",
        "",
        "- Grouped metric comparison",
        "- Attack priority metrics",
        "- Weighted score lollipop chart",
        "- ROC-AUC và PR-AUC comparison",
        "- Training time comparison",
        "- Prediction time comparison",
        "- Confusion matrix component comparison",
        "- False negative comparison",
        "- False positive comparison",
        "- Radar chart",
        "- Combined ROC curves",
        "- Combined Precision-Recall curves",
        "- Summary table image",
        "",
        "## 7. Nhận xét gợi ý",
        "",
        "Logistic Regression là baseline tuyến tính, thường có tốc độ huấn luyện và dự đoán nhanh, nhưng có thể hạn chế khi dữ liệu có quan hệ phi tuyến.",
        "Decision Tree dễ giải thích hơn nhờ feature importance và luật rẽ nhánh, nhưng có nguy cơ overfitting nếu cây quá sâu.",
        "Deep MLP có khả năng học quan hệ phi tuyến phức tạp hơn, nhưng thời gian huấn luyện thường cao hơn và khó giải thích hơn.",
        "Mô hình được chọn nên cân bằng giữa Recall/F1 của lớp Attack, thời gian dự đoán và khả năng giải thích.",
    ])

    content = "\n".join(lines)

    report_path = COMPARISON_RESULT_DIR / "model_comparison_report_notes.md"
    report_path.write_text(content, encoding="utf-8")

    print(f"[OK] Saved report notes: {report_path}")


# ============================================================
# 12. CREATE ALL FIGURES
# ============================================================

def create_all_figures(ranking_df):
    print_section("5. CREATE MODEL COMPARISON FIGURES")

    plot_grouped_metric_comparison(ranking_df)
    plot_attack_priority_metrics(ranking_df)
    plot_weighted_score_lollipop(ranking_df)
    plot_roc_auc_pr_auc_comparison(ranking_df)

    plot_training_time_comparison(ranking_df)
    plot_prediction_time_comparison(ranking_df)
    plot_prediction_time_per_sample(ranking_df)

    plot_confusion_components(ranking_df)
    plot_false_negative_comparison(ranking_df)
    plot_false_positive_comparison(ranking_df)

    plot_radar_chart(ranking_df)
    plot_combined_roc_curves()
    plot_combined_precision_recall_curves()
    plot_summary_table(ranking_df)


# ============================================================
# 13. SAVE FINAL SUMMARY
# ============================================================

def save_final_summary(ranking_df):
    print_section("6. SAVE FINAL SUMMARY")

    best_model = ranking_df.iloc[0]

    final_summary = {
        "best_model_by_weighted_score": best_model["display_name"],
        "best_model_weighted_score": float(best_model["weighted_score"]),
        "ranking_order": ranking_df["display_name"].tolist(),
        "selection_reason": (
            "The selected model is ranked highest by a weighted score that prioritizes "
            "Attack Recall and Attack F1-score, which are important for IDS tasks."
        ),
    }

    save_json(final_summary, COMPARISON_RESULT_DIR / "final_model_selection_summary.json")

    print("Best model by weighted score:")
    print(f"- {best_model['display_name']}: {best_model['weighted_score']:.4f}")


# ============================================================
# 14. MAIN
# ============================================================

def main():
    check_required_model_outputs()

    comparison_df = load_model_metrics()

    ranking_full_df, ranking_df = compute_model_ranking(comparison_df)

    ranking_full_df.to_csv(
        COMPARISON_RESULT_DIR / "model_metrics_comparison_with_ranking.csv",
        index=False,
    )

    create_all_figures(ranking_df)

    generate_report_notes(ranking_df)

    save_final_summary(ranking_df)

    print("\nModel comparison completed successfully.")
    print(f"Comparison figures saved to: {COMPARISON_FIGURE_DIR}")
    print(f"Comparison results saved to: {COMPARISON_RESULT_DIR}")


if __name__ == "__main__":
    main()