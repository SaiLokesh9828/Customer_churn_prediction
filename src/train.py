"""
train.py

Trains and compares multiple models for churn prediction:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Each non-baseline model is tuned with RandomizedSearchCV (5-fold
stratified CV, scored on F1 since the dataset is moderately imbalanced
~27% positive class). The best model overall (by validation F1) is
saved along with the fitted preprocessor and a metrics report.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_preprocessing import load_config
from feature_engineering import (
    engineer_features, build_preprocessor, get_feature_lists, save_preprocessor
)


def load_split(path: str, target_column: str):
    df = pd.read_csv(path)
    y = df[target_column].values
    X = df.drop(columns=[target_column])
    return X, y


def prepare_features(X: pd.DataFrame, config: dict):
    X = engineer_features(X)
    id_col = config["features"]["id_column"]
    if id_col in X.columns:
        X = X.drop(columns=[id_col])
    numeric_columns, categorical_columns = get_feature_lists(config)
    return X, numeric_columns, categorical_columns


def get_search_spaces():
    return {
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(3, 20),
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                random_state=42, eval_metric="logloss", use_label_encoder=False
            ),
            "params": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(2, 10),
                "learning_rate": uniform(0.01, 0.29),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "scale_pos_weight": uniform(1, 3),  # handles class imbalance
            },
        },
        "lightgbm": {
            "model": LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1),
            "params": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(2, 15),
                "learning_rate": uniform(0.01, 0.29),
                "num_leaves": randint(15, 100),
                "subsample": uniform(0.6, 0.4),
            },
        },
    }


def train_baseline(X_train, y_train, preprocessor):
    """Simple, un-tuned Logistic Regression baseline for comparison."""
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    X_train_transformed = preprocessor.fit_transform(X_train)
    baseline.fit(X_train_transformed, y_train)
    return baseline


def tune_model(name, spec, X_train_transformed, y_train, n_iter=25, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=spec["model"],
        param_distributions=spec["params"],
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train_transformed, y_train)
    print(f"[{name}] best CV F1: {search.best_score_:.4f} | best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


def evaluate(model, X_val_transformed, y_val, name):
    preds = model.predict(X_val_transformed)
    probs = model.predict_proba(X_val_transformed)[:, 1]
    f1 = f1_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    print(f"\n[{name}] Validation F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_val, preds, target_names=["No Churn", "Churn"]))
    return {"f1": f1, "roc_auc": auc}


def main():
    config = load_config()
    target_column = config["data"]["target_column"]

    X_train, y_train = load_split("data/processed/train.csv", target_column)
    X_val, y_val = load_split("data/processed/val.csv", target_column)

    X_train, numeric_cols, categorical_cols = prepare_features(X_train, config)
    X_val, _, _ = prepare_features(X_val, config)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit preprocessor ONLY on training data; transform val with same fit.
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    results = {}
    fitted_models = {}

    # --- Baseline ---
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train_t, y_train)
    results["logistic_regression_baseline"] = evaluate(
        baseline, X_val_t, y_val, "Logistic Regression (baseline)"
    )
    fitted_models["logistic_regression_baseline"] = baseline

    # --- Tuned models ---
    search_spaces = get_search_spaces()
    for name, spec in search_spaces.items():
        best_estimator, cv_f1 = tune_model(
            name, spec, X_train_t, y_train,
            n_iter=config["training"]["n_trials_optuna"],
            cv_folds=config["training"]["cv_folds"],
        )
        results[name] = evaluate(best_estimator, X_val_t, y_val, name)
        results[name]["cv_f1"] = cv_f1
        fitted_models[name] = best_estimator

    # --- Select best model by validation F1 ---
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted_models[best_name]
    print(f"\n=== Best model: {best_name} (Val F1={results[best_name]['f1']:.4f}) ===")

    os.makedirs(config["training"]["models_dir"], exist_ok=True)
    joblib.dump(best_model, config["paths"]["best_model_path"])
    save_preprocessor(preprocessor, config["paths"]["preprocessor_path"])

    with open(config["paths"]["metrics_path"], "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    feature_names = list(preprocessor.get_feature_names_out())
    with open(config["paths"]["feature_names_path"], "w") as f:
        json.dump(feature_names, f, indent=2)

    print("Saved best model, preprocessor, metrics, and feature names to models/.")


if __name__ == "__main__":
    main()
