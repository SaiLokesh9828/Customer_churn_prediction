"""
data_preprocessing.py

Loads the raw Telco Churn CSV, cleans it, and performs an initial
train/val/test split. No feature engineering happens here (see
feature_engineering.py) — this module is only responsible for loading,
type-fixing, and leakage-safe splitting.
"""

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{raw_path}'. Download the Telco "
            f"Customer Churn CSV from Kaggle (blastchar/telco-customer-churn) "
            f"and place it at this path. See data/README.md."
        )
    df = pd.read_csv(raw_path)
    return df


def clean_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Fix known data-quality issues in the Telco dataset."""
    df = df.copy()

    # TotalCharges is stored as object with some blank strings for
    # customers with tenure == 0 (new customers). Convert and impute.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_mask = df["TotalCharges"].isna()
    if missing_mask.any():
        # For brand-new customers (tenure == 0), TotalCharges is
        # legitimately 0, not missing at random.
        df.loc[missing_mask & (df["tenure"] == 0), "TotalCharges"] = 0.0
        # Any remaining missing values are imputed with the median as a
        # safe fallback (logged, not silently dropped).
        remaining = df["TotalCharges"].isna().sum()
        if remaining > 0:
            median_val = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    # Normalize the target to binary 0/1.
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    # SeniorCitizen is 0/1 already but stored as int; keep as-is,
    # feature_engineering.py will decide how to encode it.

    # Drop exact duplicate rows if any.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")

    return df


def split_data(df: pd.DataFrame, target_column: str, test_size: float,
                val_size: float, random_state: int):
    """
    Stratified split into train/val/test. Splitting happens BEFORE any
    feature engineering that fits on data (e.g., encoders, scalers) to
    avoid data leakage.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target_column],
        random_state=random_state
    )
    # val_size is expressed as a fraction of the ORIGINAL dataset;
    # convert it to a fraction of train_val.
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size,
        stratify=train_val[target_column], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    target_column = config["data"]["target_column"]

    df = load_raw_data(raw_path)
    df = clean_data(df, target_column)

    train, val, test = split_data(
        df, target_column,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    print(f"Churn rate — train: {train[target_column].mean():.3f}, "
          f"val: {val[target_column].mean():.3f}, "
          f"test: {test[target_column].mean():.3f}")


if __name__ == "__main__":
    main()
