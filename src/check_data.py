from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

FEATURE_PATH_OPTIONS = [
    RAW_DIR / "NUSW_NB15_features.csv",
    RAW_DIR / "NUSW-NB15_features.csv",
    RAW_DIR / "UNSW_NB15_features.csv",
    RAW_DIR / "UNSW-NB15_features.csv",
]

FEATURE_PATH = next((path for path in FEATURE_PATH_OPTIONS if path.exists()), None)


def check_file_exists(path: Path, name: str):
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing {name}")
    print(f"[OK] Found {name}: {path.name}")


def main():
    print("Checking dataset files...\n")

    check_file_exists(TRAIN_PATH, "training set")
    check_file_exists(TEST_PATH, "testing set")
    check_file_exists(FEATURE_PATH, "feature description file")

    print("\nLoading CSV files...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    features_df = pd.read_csv(FEATURE_PATH, encoding="latin1")

    print("\nDataset loaded successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Features shape: {features_df.shape}")

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nFirst 5 rows of train:")
    print(train_df.head())

    print("\nMissing values in train:")
    missing_values = train_df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) == 0:
        print("No missing values found.")
    else:
        print(missing_values)

    print("\nLabel distribution in train:")
    print(train_df["label"].value_counts())

    print("\nLabel ratio in train:")
    print(train_df["label"].value_counts(normalize=True))

    if "attack_cat" in train_df.columns:
        print("\nAttack category distribution in train:")
        print(train_df["attack_cat"].value_counts())

    print("\nFirst 5 rows of feature description:")
    print(features_df.head())


if __name__ == "__main__":
    main()