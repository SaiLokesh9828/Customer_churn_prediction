"""
evaluate.py

Loads the saved best model + preprocessor and reports FINAL test-set
metrics. The test set is touched exactly once, here, after all model
selection/tuning decisions were made using train/val only — this keeps
the test score an honest, unbiased estimate of generalization.
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, f1_score, precision_score, recall_score, accuracy_score
)

from data_preprocessing import load_config
from feature_engineering import engineer_features, get_feature_lists


def prepare_features(X: pd.DataFrame, config: dict):
    X = engineer_features(X)
    id_col = config["features"]["id_column"]
    if id_col in X.columns:
        X = X.drop(columns=[id_col])
    return X


def main():
    config = load_config()
    target_column = config["data"]["target_column"]

    test_df = pd.read_csv("data/processed/test.csv")
    y_test = test_df[target_column].values
    X_test = test_df.drop(columns=[target_column])
    X_test = prepare_features(X_test, config)

    model = joblib.load(config["paths"]["best_model_path"])
    preprocessor = joblib.load(config["paths"]["preprocessor_path"])

    X_test_t = preprocessor.transform(X_test)
    preds = model.predict(X_test_t)
    probs = model.predict_proba(X_test_t)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print("=== FINAL TEST SET METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\n", classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    with open("models/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probs)
    plt.title("ROC Curve — Test Set")
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=150)
    plt.close()

    print("\nSaved confusion_matrix.png, roc_curve.png, and test_metrics.json to models/.")


if __name__ == "__main__":
    main()
