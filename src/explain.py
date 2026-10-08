"""
explain.py

Generates SHAP-based explainability artifacts for the best trained model:
  - Global feature importance (summary plot)
  - Per-prediction waterfall explanation for a sample customer

Works with tree-based models (RF/XGBoost/LightGBM) via TreeExplainer.
Falls back to LinearExplainer for logistic regression.
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

    # Load best model name
    with open(config["paths"]["metrics_path"]) as f:
        best_model_name = json.load(f)["best_model"]

    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=[target_column])
    X_test = prepare_features(X_test, config)

    # Load trained model and preprocessing objects
    model = joblib.load(config["paths"]["best_model_path"])
    preprocessor = joblib.load(config["paths"]["preprocessor_path"])

    with open(config["paths"]["feature_names_path"]) as f:
        feature_names = json.load(f)

    # Transform test data
    X_test_t = preprocessor.transform(X_test)

    # Convert sparse matrix if necessary
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    # Manageable sample for SHAP
    sample = X_test_t[:200]

    sample_df = pd.DataFrame(
        sample,
        columns=feature_names
    )

    # ---------------------------------------------------------
    # Create SHAP explainer
    # ---------------------------------------------------------

    if best_model_name == "logistic_regression_baseline":
        explainer = shap.LinearExplainer(model, sample_df)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample_df)

   # Handle SHAP output for binary/multi-output classifiers
    if isinstance(shap_values, list):
     shap_values = shap_values[1]

    if isinstance(shap_values, shap.Explanation):
     shap_values = shap_values.values

   # Newer SHAP versions may return:
   # (samples, features, classes)
   # Select class 1 = churn
    if shap_values.ndim == 3:
     shap_values = shap_values[:, :, 1]

    print("SHAP values shape:", shap_values.shape)

    # ---------------------------------------------------------
    # GLOBAL SHAP SUMMARY
    # ---------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        sample_df,
        show=False,
        max_display=15
    )

    plt.tight_layout()

    plt.savefig(
        "models/shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # SINGLE CUSTOMER SHAP EXPLANATION
    # ---------------------------------------------------------

    # Convert expected value to a scalar
    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).flatten()[0]

    # Create SHAP Explanation object
    single_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=sample_df.iloc[0].values,
        feature_names=feature_names
    )

    # Waterfall plot is more appropriate for one prediction
    plt.figure()

    shap.plots.waterfall(
        single_explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "models/shap_single_customer.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved shap_summary.png and shap_single_customer.png to models/.")


if __name__ == "__main__":
    main()
