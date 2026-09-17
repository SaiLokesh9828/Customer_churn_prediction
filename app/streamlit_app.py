"""
streamlit_app.py

Interactive Streamlit app for the Customer Churn Prediction model.
Run with:  streamlit run app/streamlit_app.py
"""

import json
import sys
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features  # noqa: E402

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
METRICS_PATH = "models/metrics.json"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, preprocessor, metrics


def main():
    st.title("📉 Customer Churn Predictor")
    st.write(
        "Predicts the likelihood that a telecom customer will churn, "
        "based on account details and subscribed services."
    )

    model, preprocessor, metrics = load_artifacts()
    if model is None:
        st.warning(
            "No trained model found at `models/best_model.pkl`. "
            "Run `python src/data_preprocessing.py`, then `python src/train.py` "
            "from the project root first."
        )
        return

    if metrics:
        best_name = metrics.get("best_model", "unknown")
        best_f1 = metrics["results"].get(best_name, {}).get("f1")
        st.caption(f"Active model: **{best_name}** — validation F1: "
                   f"{best_f1:.3f}" if best_f1 else f"Active model: **{best_name}**")

    st.divider()
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"]
        )

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))

    if st.button("Predict Churn Risk", type="primary"):
        input_dict = {
            "customerID": ["N/A"],
            "gender": [gender], "SeniorCitizen": [senior_citizen],
            "Partner": [partner], "Dependents": [dependents], "tenure": [tenure],
            "PhoneService": [phone_service], "MultipleLines": [multiple_lines],
            "InternetService": [internet_service], "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup], "DeviceProtection": [device_protection],
            "TechSupport": [tech_support], "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies], "Contract": [contract],
            "PaperlessBilling": [paperless_billing], "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges], "TotalCharges": [total_charges],
        }
        input_df = pd.DataFrame(input_dict)
        input_df = engineer_features(input_df)
        input_df = input_df.drop(columns=["customerID"])

        X_t = preprocessor.transform(input_df)
        pred = model.predict(X_t)[0]
        prob = model.predict_proba(X_t)[0, 1]

        st.divider()
        st.subheader("Result")
        if pred == 1:
            st.error(f"⚠️ High churn risk — predicted probability: **{prob:.1%}**")
        else:
            st.success(f"✅ Low churn risk — predicted probability: **{prob:.1%}**")

        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            "This probability reflects the model's confidence, not a guarantee. "
            "Use alongside business judgment for retention decisions."
        )


if __name__ == "__main__":
    main()
