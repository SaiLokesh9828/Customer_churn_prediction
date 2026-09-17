"""
feature_engineering.py

Builds derived features and a fit-on-train-only preprocessing pipeline
(ColumnTransformer) to avoid data leakage. The fitted pipeline is saved
so the exact same transformation can be applied at inference time.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven derived features. Pure function of a single row's
    existing columns — safe to apply identically to train/val/test/inference."""
    df = df.copy()

    # Tenure buckets capture non-linear churn risk by customer lifecycle stage.
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"]
    ).astype(str)

    # Average monthly spend relative to tenure — flags customers whose
    # charges have grown or shrunk sharply, a common churn precursor.
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["charge_deviation"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # Count of subscribed add-on services — a proxy for account "stickiness".
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # Contract risk flag — month-to-month customers churn far more often.
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Paperless + electronic check has historically been a strong churn
    # signal in this dataset; keep as separate interaction feature.
    df["electronic_check_paperless"] = (
        (df["PaymentMethod"] == "Electronic check") &
        (df["PaperlessBilling"] == "Yes")
    ).astype(int)

    return df


def build_preprocessor(numeric_columns, categorical_columns) -> ColumnTransformer:
    """ColumnTransformer fit ONLY on training data (fit_transform in train.py)."""
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])
    return preprocessor


def get_feature_lists(config: dict):
    """Return updated numeric/categorical lists including engineered features."""
    numeric_columns = config["features"]["numeric_columns"] + [
        "avg_monthly_spend", "charge_deviation", "num_services"
    ]
    categorical_columns = config["features"]["categorical_columns"] + [
        "tenure_group"
    ]
    # is_month_to_month / electronic_check_paperless are already binary
    # 0/1 — treat as numeric so they aren't one-hot re-expanded.
    numeric_columns += ["is_month_to_month", "electronic_check_paperless"]
    return numeric_columns, categorical_columns


def save_preprocessor(preprocessor, path: str):
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str):
    return joblib.load(path)
