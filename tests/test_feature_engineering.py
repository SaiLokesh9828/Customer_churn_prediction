"""
test_feature_engineering.py

Lightweight unit tests for the feature engineering functions.
Run with: pytest tests/
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features  # noqa: E402


def make_sample_row():
    return pd.DataFrame({
        "tenure": [24],
        "MonthlyCharges": [70.0],
        "TotalCharges": [1680.0],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["Yes"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaymentMethod": ["Electronic check"],
        "PaperlessBilling": ["Yes"],
    })


def test_num_services_count():
    df = engineer_features(make_sample_row())
    assert df.loc[0, "num_services"] == 3  # OnlineSecurity, DeviceProtection, StreamingTV


def test_is_month_to_month_flag():
    df = engineer_features(make_sample_row())
    assert df.loc[0, "is_month_to_month"] == 1


def test_electronic_check_paperless_flag():
    df = engineer_features(make_sample_row())
    assert df.loc[0, "electronic_check_paperless"] == 1


def test_avg_monthly_spend_no_division_by_zero():
    df = make_sample_row()
    df["tenure"] = 0
    result = engineer_features(df)
    assert result.loc[0, "avg_monthly_spend"] == result.loc[0, "TotalCharges"]  # divided by 1, not 0
