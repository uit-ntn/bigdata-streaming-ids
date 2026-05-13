# BigData Streaming IDS

A Big Data and Machine Learning project for near real-time Network Intrusion Detection using the UNSW-NB15 dataset.

## Overview

`bigdata-streaming-ids` builds a complete IDS pipeline for detecting whether network traffic is **Normal** or **Attack**. The project uses the UNSW-NB15 dataset, applies a reusable preprocessing pipeline, trains multiple machine learning/deep learning models, compares model performance, and prepares the system for micro-batch streaming simulation.

The project is designed for a Big Data course and focuses on both model performance and streaming-style processing.

## Main Objectives

- Explore the Network Intrusion Detection System (IDS) problem.
- Analyze the UNSW-NB15 dataset and attack categories.
- Build detailed EDA visualizations and summary reports.
- Preprocess numerical and categorical features.
- Train and evaluate three models:
  - Logistic Regression
  - Decision Tree
  - Deep MLP
- Compare models using IDS-focused metrics.
- Prioritize **Recall** and **F1-score** for the Attack class.
- Prepare report-ready outputs and demo materials.
- Support later micro-batch streaming simulation.

## Dataset

This project uses the **UNSW-NB15** dataset.

Required files:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
NUSW_NB15_features.csv
```

Depending on the downloaded filename, the feature file may also be named:

```text
NUSW-NB15_features.csv
UNSW_NB15_features.csv
UNSW-NB15_features.csv
```

Place the dataset files in:

```text
data/raw/
```

Expected structure:

```text
bigdata-streaming-ids/
└── data/
    └── raw/
        ├── UNSW_NB15_training-set.csv
        ├── UNSW_NB15_testing-set.csv
        └── NUSW_NB15_features.csv
```

Raw and processed datasets are not committed to GitHub because of file size and reproducibility concerns.

## Project Structure

```text
bigdata-streaming-ids/
│
├── data/
│   ├── raw/                         # Original UNSW-NB15 dataset files
│   └── processed/                   # Processed matrices and labels
│
├── models/
│   ├── logistic_regression/         # Logistic Regression model
│   ├── decision_tree/               # Decision Tree model
│   └── deep_mlp/                    # Deep Learning MLP model
│
├── reports/
│   ├── figures/
│   │   ├── eda/                     # EDA visualizations
│   │   ├── preprocessing/           # Preprocessing visualizations
│   │   ├── models/
│   │   │   ├── logistic_regression/
│   │   │   ├── decision_tree/
│   │   │   └── deep_mlp/
│   │   └── model_comparison/        # Comparison charts for all models
│   │
│   └── results/
│       ├── eda/                     # EDA summary CSV/JSON/notes
│       ├── preprocessing/           # Preprocessing summary CSV/JSON/notes
│       ├── models/
│       │   ├── logistic_regression/
│       │   ├── decision_tree/
│       │   └── deep_mlp/
│       └── model_comparison/        # Final model comparison outputs
│
├── src/
│   ├── check_data.py                # Check dataset paths and basic loading
│   ├── eda.py                       # Detailed exploratory data analysis
│   ├── preprocessing.py             # Reusable preprocessing pipeline
│   ├── train_logistic_regression.py # Train Logistic Regression model
│   ├── train_decision_tree.py       # Train Decision Tree model
│   ├── train_deep_mlp.py            # Train Deep Learning MLP model
│   └── compare_models.py            # Compare all trained models
│
├── notebooks/                       # Optional notebooks
├── docs/                            # Project documents
├── requirements.txt
├── README.md
└── .gitignore
```

## Methodology

### 1. Data Checking

The script `src/check_data.py` verifies whether the required dataset files are placed correctly in `data/raw/`.

It checks:

- Training set file.
- Testing set file.
- Feature description file.
- Dataset shape.
- Columns.
- Label distribution.
- Attack category distribution.

Run:

```bash
python src/check_data.py
```

### 2. Exploratory Data Analysis

The script `src/eda.py` performs detailed exploratory data analysis and saves both figures and tables.

It includes:

- Dataset size comparison.
- Missing value analysis.
- Duplicate row checking.
- Label distribution.
- Attack category distribution.
- Categorical feature analysis.
- Numerical feature analysis.
- Histograms and log histograms.
- Boxplots by label.
- Scatter plot samples.
- Feature mean difference by label.
- Correlation analysis.
- Train/test difference summary.

Run:

```bash
python src/eda.py
```

Outputs:

```text
reports/figures/eda/
reports/results/eda/
```

### 3. Preprocessing

The script `src/preprocessing.py` prepares data for model training.

Main steps:

- Replace infinite values with missing values.
- Drop unused columns: `id`, `attack_cat`.
- Split features and target label.
- Identify categorical and numerical columns.
- Impute missing numerical values using median.
- Impute missing categorical values using the most frequent value.
- Scale numerical columns using `StandardScaler`.
- Encode categorical columns using `OneHotEncoder`.
- Save reusable preprocessor.
- Save processed train/test matrices.

Run:

```bash
python src/preprocessing.py
```

Outputs:

```text
data/processed/
├── X_train_processed.npz
├── X_test_processed.npz
├── y_train.npy
├── y_test.npy
├── column_info.json
└── feature_names.json
```

```text
models/
└── preprocessor.joblib
```

```text
reports/figures/preprocessing/
reports/results/preprocessing/
```

### 4. Model Training

Each model is trained in a separate script. Each model has its own folder for saved models, figures, and results.

#### Model 1: Logistic Regression

Logistic Regression is used as a linear baseline model.

```bash
python src/train_logistic_regression.py
```

Outputs:

```text
models/logistic_regression/
reports/figures/models/logistic_regression/
reports/results/models/logistic_regression/
```

#### Model 2: Decision Tree

Decision Tree is used as a non-linear baseline model and provides feature importance and tree interpretability.

```bash
python src/train_decision_tree.py
```

Outputs:

```text
models/decision_tree/
reports/figures/models/decision_tree/
reports/results/models/decision_tree/
```

#### Model 3: Deep MLP

Deep MLP is used as the Deep Learning model for tabular feature-vector data.

```bash
python src/train_deep_mlp.py
```

Outputs:

```text
models/deep_mlp/
reports/figures/models/deep_mlp/
reports/results/models/deep_mlp/
```

### 5. Model Comparison

The script `src/compare_models.py` compares the three trained models.

It reads:

```text
reports/results/models/logistic_regression/metrics_summary.csv
reports/results/models/decision_tree/metrics_summary.csv
reports/results/models/deep_mlp/metrics_summary.csv
```

It compares models using:

- Accuracy
- Precision for Attack class
- Recall for Attack class
- F1-score for Attack class
- ROC-AUC
- PR-AUC
- Training time
- Prediction time
- False positives
- False negatives
- Weighted model score

Run:

```bash
python src/compare_models.py
```

Outputs:

```text
reports/figures/model_comparison/
reports/results/model_comparison/
```

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision Attack | Among predicted attacks, how many are truly attacks |
| Recall Attack | Among actual attacks, how many are detected |
| F1 Attack | Balance between Precision and Recall for Attack class |
| ROC-AUC | Ranking quality across thresholds |
| PR-AUC | Precision-Recall performance, useful for imbalanced data |
| Confusion Matrix | TN, FP, FN, TP breakdown |

For IDS, **Recall Attack** and **F1 Attack** are prioritized because missing attacks can create serious security risks.

## Model Selection Score

The comparison script computes a weighted score:

```text
Weighted Score =
0.35 * Recall_Attack
+ 0.30 * F1_Attack
+ 0.15 * Precision_Attack
+ 0.10 * ROC_AUC
+ 0.10 * PR_AUC
```

This formula gives higher importance to detecting attacks correctly.

## Installation

Clone the repository:

```bash
git clone https://github.com/uit-ntn/bigdata-streaming-ids.git
cd bigdata-streaming-ids
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If TensorFlow is not installed, install it separately:

```bash
pip install tensorflow
```

## Recommended Requirements

Example `requirements.txt`:

```text
pandas
numpy
scipy
scikit-learn
matplotlib
joblib
tensorflow
jupyter
```

## Full Running Order

Run the project in this order:

```bash
python src/check_data.py
python src/eda.py
python src/preprocessing.py
python src/train_logistic_regression.py
python src/train_decision_tree.py
python src/train_deep_mlp.py
python src/compare_models.py
```

## Expected Outputs

After running the full pipeline, expected outputs include:

- EDA charts and result tables.
- Preprocessing pipeline and visualizations.
- Processed sparse matrices for train/test.
- Trained Logistic Regression model.
- Trained Decision Tree model.
- Trained Deep MLP model.
- Metrics for each model.
- Confusion matrices.
- ROC and Precision-Recall curves.
- Feature importance charts.
- Training history charts for Deep MLP.
- Final model comparison tables and visualizations.
- Report notes in Markdown format for each stage.

## Report-ready Notes

Several scripts automatically generate Markdown notes for writing the final report:

```text
reports/results/eda/eda_report_notes.md
reports/results/preprocessing/preprocessing_report_notes.md
reports/results/models/logistic_regression/logistic_regression_report_notes.md
reports/results/models/decision_tree/decision_tree_report_notes.md
reports/results/models/deep_mlp/deep_mlp_report_notes.md
reports/results/model_comparison/model_comparison_report_notes.md
```

These files can be used as draft content for the final project report.

## Streaming Simulation

The current pipeline prepares the project for micro-batch streaming simulation. After selecting the best model, the testing set can be split into small batches to simulate incoming network traffic.

For each micro-batch, the system can record:

- Batch ID.
- Number of records.
- Number of predicted attacks.
- Prediction time.
- Processing latency.
- Batch-level detection summary.

This part can be implemented in pure Python first and extended to Spark Structured Streaming if required.

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Joblib
- Jupyter Notebook
- Optional: Apache Spark / Spark Structured Streaming

## Notes

This project does not deploy an IDS directly on a real production network. It uses a public dataset and simulates near real-time detection through micro-batch processing.

Raw dataset files, processed datasets, and trained model artifacts may be ignored by Git depending on `.gitignore` configuration.

## Author

Nguyen Thanh Nhan

University of Information Technology  
Faculty of Information Systems  
Course: Big Data

## License

This project is intended for academic and learning purposes.
