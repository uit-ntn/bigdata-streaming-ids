# BigData Network IDS

A machine learning and big data streaming project for real-time network intrusion detection using the UNSW-NB15 dataset.

## Overview

This project builds a Network Intrusion Detection System (IDS) pipeline that detects whether network traffic is normal or malicious. The system uses machine learning models trained on the UNSW-NB15 dataset and simulates near real-time detection through micro-batch processing.

The project is designed for a Big Data course project and focuses on both model performance and streaming-style processing.

## Objectives

- Explore the network intrusion detection problem and common attack categories.
- Prepare and preprocess the UNSW-NB15 dataset.
- Train baseline and advanced machine learning models for binary classification.
- Evaluate models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
- Prioritize Recall and F1-score for the attack class.
- Simulate real-time detection using micro-batches.
- Analyze feature importance to explain model predictions.

## Dataset

This project uses the UNSW-NB15 dataset.

Required files:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
UNSW_NB15_features.csv
```

The dataset should be placed in:

```text
data/raw/
```

The raw dataset files are not committed to GitHub because of file size and reproducibility concerns.

## Project Structure

```text
bigdata-network-ids/
│
├── data/
│   ├── raw/                 # Original dataset files
│   ├── processed/           # Processed datasets
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_models.ipynb
│   └── 04_micro_batch_streaming.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── streaming_simulation.py
│   └── utils.py
│
├── models/                  # Saved trained models
├── reports/
│   ├── figures/             # Charts and visualizations
│   └── results/             # Evaluation result files
│
├── docs/                    # Project documents
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Methodology

### 1. Data Preparation

The dataset is loaded from CSV files. Unnecessary columns such as IDs are removed, missing values are handled, and the target label is prepared for binary classification.

### 2. Preprocessing

Categorical features such as protocol, service, and state are encoded. Numerical features are scaled when required. The preprocessing pipeline is designed to be reused for both training data and micro-batch prediction data.

### 3. Model Training

The project starts with baseline models and then compares stronger machine learning models.

Planned models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost or LightGBM

### 4. Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

For intrusion detection, Recall for the attack class is especially important because missing an attack can create serious security risk.

### 5. Micro-batch Streaming Simulation

After selecting a suitable model, the testing set is split into small batches. Each batch simulates incoming network traffic.

For each batch, the system records:

- Number of predicted attacks
- Prediction time
- Processing latency
- Batch-level detection summary

### 6. Model Explanation

Feature importance from tree-based models or a basic SHAP analysis can be used to identify which network traffic features contribute most to model decisions.

## Installation

Clone the repository:

```bash
git clone https://github.com/uit-ntn/bigdata-network-ids.git
cd bigdata-network-ids
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
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

## Basic Usage

Run exploratory data analysis:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Train models:

```bash
python src/train.py
```

Evaluate models:

```bash
python src/evaluate.py
```

Run micro-batch streaming simulation:

```bash
python src/streaming_simulation.py
```

## Expected Outputs

- Preprocessed training and testing data.
- Trained machine learning models.
- Evaluation metrics table.
- Confusion matrix visualization.
- Feature importance chart.
- Micro-batch latency results.
- Final report and demo materials.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost or LightGBM
- Matplotlib
- Seaborn
- Jupyter Notebook
- Optional: Apache Spark / Spark Structured Streaming

## Notes

This project does not deploy an IDS directly on a real production network. It uses a public dataset and simulates streaming behavior to demonstrate the feasibility of near real-time intrusion detection.

## Author

Nguyen Thanh Nhan

University of Information Technology  
Faculty of Information Systems  
Course: Big Data

## License

This project is intended for academic and learning purposes.
