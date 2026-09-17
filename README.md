# 📉 Customer Churn Prediction

An end-to-end machine learning pipeline that predicts telecom customer churn,
compares multiple algorithms, and explains individual predictions with SHAP.
Deployed as an interactive Streamlit app.

## Overview

Customer acquisition costs 5-25x more than retention. This project builds a
model that flags at-risk customers *before* they cancel, so a retention team
can intervene early, and explains *why* each customer is flagged so the
outreach can be targeted (e.g., offer a contract upgrade vs. a discount).

**Real-world use case:** A telecom's customer success team runs this model
weekly against active accounts and prioritizes outreach by churn probability.

## Features

- Leakage-safe preprocessing (train/val/test split before any fitting)
- Domain-driven feature engineering (tenure buckets, service count, spend deviation)
- Baseline (Logistic Regression) vs. tuned Random Forest, XGBoost, LightGBM
- Class-imbalance handling (`class_weight`, `scale_pos_weight`)
- Hyperparameter tuning via `RandomizedSearchCV` with stratified 5-fold CV
- Model selection by validation F1, final unbiased evaluation on held-out test set
- SHAP-based global and per-customer explainability
- Interactive Streamlit app for single-customer predictions

## Dataset

IBM Telco Customer Churn dataset (public, Kaggle: `blastchar/telco-customer-churn`),
~7,043 customers, 21 columns. See `data/README.md` for setup instructions.

## Architecture / Pipeline

```
Raw CSV
   │
   ▼
data_preprocessing.py  →  clean types, fix TotalCharges blanks, stratified train/val/test split
   │
   ▼
feature_engineering.py →  derived features + ColumnTransformer (StandardScaler + OneHotEncoder)
   │
   ▼
train.py                →  Logistic Regression baseline + tuned RF / XGBoost / LightGBM
   │                        (RandomizedSearchCV, 5-fold stratified CV, F1-scored)
   ▼
evaluate.py              →  final test-set metrics, confusion matrix, ROC curve
   │
   ▼
explain.py                →  SHAP global + per-customer explanations
   │
   ▼
app/streamlit_app.py       →  interactive prediction UI
```

## Project Structure

```
customer-churn-prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                # place telco_churn.csv here
│   ├── processed/          # generated: train/val/test.csv
│   └── README.md
├── models/                 # generated: best_model.pkl, preprocessor.pkl, metrics.json,
│                            #            confusion_matrix.png, roc_curve.png, shap_*.png
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── tests/
│   └── test_feature_engineering.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd customer-churn-prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset per `data/README.md` and place it at `data/raw/telco_churn.csv`.

## Usage

```bash
python src/data_preprocessing.py   # clean + split
python src/train.py                # train, tune, compare, save best model
python src/evaluate.py             # final test-set metrics + plots
python src/explain.py              # SHAP explainability artifacts
streamlit run app/streamlit_app.py # launch the prediction app
pytest tests/                      # run unit tests
```

## Screenshots

`[Add: app/screenshots/prediction_form.png]`
`[Add: app/screenshots/prediction_result.png]`
`[Add: models/confusion_matrix.png]`
`[Add: models/shap_summary.png]`

## Results

Exact metrics depend on the random seed and the specific Kaggle CSV revision
used. After running the pipeline, results are written to
`models/metrics.json` (validation) and `models/test_metrics.json` (final
test set). Report your own run's numbers here, e.g.:

- Best model: `[BEST_MODEL_NAME]`
- Test F1: `[TEST_F1]`
- Test ROC-AUC: `[TEST_ROC_AUC]`

*(Placeholders are intentional — this README does not fabricate results.)*

## Future Improvements

- Add SMOTE/ADASYN comparison against class-weighting for imbalance handling
- Track experiments with MLflow instead of a single `metrics.json`
- Add a FastAPI backend for batch scoring alongside the Streamlit UI
- Calibrate probabilities (Platt scaling) if used for threshold-based alerting
- Add monitoring for feature/label drift in production

## Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib ·
Seaborn · Streamlit · joblib · PyYAML

## Author

`[Your Name]` — `[email / LinkedIn / GitHub]`
