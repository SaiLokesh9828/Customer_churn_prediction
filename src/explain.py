"""
explain.py

Generates SHAP-based explainability artifacts for the best trained model:
  - Global feature importance (summary plot)
  - Per-prediction force/waterfall explanation for a sample customer

Works out of the box for tree-based models (RF/XGBoost/LightGBM) via
TreeExplainer. Falls back to KernelExplainer for linear models (slower).
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from data_preprocessing import load_config
from feature_engineering import engineer_features


def prepare_features(X: pd.DataFrame, config: dict):
    X = engineer_features(X)
    id_col = config["features"]["id_column"]
    if id_col in X.columns:
        X = X.drop(columns=[id_col])
    return X


def main():
    config = load_config()
    target_column = config["data"]["target_column"]

    with open(config["paths"]["metrics_path"]) as f:
        best_model_name = json.load(f)["best_model"]

    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=[target_column])
    X_test = prepare_features(X_test, config)

    model = joblib.load(config["paths"]["best_model_path"])
    preprocessor = joblib.load(config["paths"]["preprocessor_path"])
    feature_names = json.load(open(config["paths"]["feature_names_path"]))

    X_test_t = preprocessor.transform(X_test)
    # Use a manageable background/explain sample for speed.
    sample = X_test_t[:200]
    sample_df = pd.DataFrame(sample, columns=feature_names)

    if best_model_name == "logistic_regression_baseline":
        explainer = shap.LinearExplainer(model, sample_df)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample_df)
    # For binary classifiers some libraries return a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample_df, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Single-customer explanation for the first row in the sample.
    plt.figure()
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0], base_values=explainer.expected_value,
            data=sample_df.iloc[0], feature_names=feature_names
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig("models/shap_single_customer.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved shap_summary.png and shap_single_customer.png to models/.")


if __name__ == "__main__":
    main()
