# \# 📉 Customer Churn Prediction

# 

# An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn and provides an interpretable churn-risk score.

# 

# The project covers the complete ML lifecycle:

# 

# \*\*Data preprocessing → Feature engineering → Model training → Hyperparameter tuning → Evaluation → SHAP explainability → Streamlit deployment\*\*

# 

# \---

# 

# \## 🚀 Live Project

# 

# The application is built using \*\*Streamlit\*\* and allows users to enter customer account and service details to receive a churn-risk prediction.

# 

# \### Prediction Output

# 

# The application provides:

# 

# \* Churn probability

# \* Low / Medium / High risk interpretation

# \* Model confidence

# \* Customer-level prediction

# 

# > The predicted probability is a model estimate and should be used alongside business judgment for retention decisions.

# 

# \---

# 

# \## 🎯 Problem Statement

# 

# Customer churn is a major challenge for telecom companies.

# 

# The objective of this project is to build a machine learning system that can:

# 

# 1\. Predict whether a customer is likely to churn.

# 2\. Estimate the probability of churn.

# 3\. Identify important factors influencing churn.

# 4\. Provide an easy-to-use interface for customer-level predictions.

# 

# This can help businesses identify high-risk customers and design targeted retention strategies.

# 

# \---

# 

# \## 📊 Dataset

# 

# The project uses the \*\*IBM Telco Customer Churn dataset\*\*.

# 

# The dataset contains customer demographic information, account information, subscribed services, contract details, payment methods, and churn labels.

# 

# \### Important Features

# 

# Examples include:

# 

# \* Gender

# \* Senior Citizen

# \* Partner

# \* Dependents

# \* Tenure

# \* Phone Service

# \* Internet Service

# \* Online Security

# \* Online Backup

# \* Device Protection

# \* Tech Support

# \* Streaming Services

# \* Contract

# \* Paperless Billing

# \* Payment Method

# \* Monthly Charges

# \* Total Charges

# 

# \### Target

# 

# `Churn`

# 

# Where:

# 

# \* `Yes` → Customer churned

# \* `No` → Customer stayed

# 

# \---

# 

# \## 🧠 Machine Learning Approach

# 

# \### 1. Data Preprocessing

# 

# The preprocessing pipeline handles:

# 

# \* Missing values

# \* Categorical variables

# \* Numerical variables

# \* Data type conversion

# \* Train/validation/test splitting

# 

# The preprocessing steps are implemented using a reusable preprocessing pipeline.

# 

# \---

# 

# \### 2. Feature Engineering

# 

# Additional features are created from the original customer information to improve model performance.

# 

# The feature engineering pipeline is applied consistently during both training and prediction.

# 

# \---

# 

# \### 3. Models

# 

# The following models are evaluated:

# 

# | Model               | Purpose                           |

# | ------------------- | --------------------------------- |

# | Logistic Regression | Baseline model                    |

# | Random Forest       | Non-linear ensemble model         |

# | XGBoost             | Gradient boosting model           |

# | LightGBM            | Efficient gradient boosting model |

# 

# \---

# 

# \## ⚙️ Hyperparameter Tuning

# 

# Model hyperparameters are optimized using:

# 

# \*\*RandomizedSearchCV\*\*

# 

# Cross-validation is used during model selection to reduce the risk of overfitting to a single validation split.

# 

# \---

# 

# \## 📈 Model Performance

# 

# The currently selected model is:

# 

# \### 🌲 Random Forest

# 

# \*\*Validation F1 Score: 0.661\*\*

# 

# The model was selected based on validation performance and is used by the Streamlit application for customer-level predictions.

# 

# > Model performance can vary depending on preprocessing, random seeds, and the training configuration.

# 

# \---

# 

# \## 🔍 Model Explainability

# 

# The project uses \*\*SHAP (SHapley Additive exPlanations)\*\* to understand model predictions.

# 

# Two explainability artifacts are generated:

# 

# \### Global Feature Importance

# 

# `models/shap\_summary.png`

# 

# Shows which features have the greatest overall influence on churn predictions.

# 

# \### Individual Customer Explanation

# 

# `models/shap\_single\_customer.png`

# 

# Shows how individual features contribute to the prediction for a specific customer.

# 

# This makes the model more interpretable and useful for business decision-making.

# 

# \---

# 

# \## 🖥️ Streamlit Application

# 

# The project includes an interactive Streamlit application.

# 

# Users can enter customer information such as:

# 

# \* Demographics

# \* Tenure

# \* Contract

# \* Internet service

# \* Payment method

# \* Monthly charges

# \* Additional subscribed services

# 

# The application then generates a churn-risk prediction.

# 

# \### Example

# 

# ```text

# Predicted churn probability: 34.9%

# 

# Risk Level: Low churn risk

# ```

# 

# The probability represents the model's estimated likelihood, not a guarantee of customer behavior.

# 

# \---

# 

# \# 📁 Project Structure

# 

# ```text

# customer-churn-prediction/

# │

# ├── app/

# │   └── streamlit\_app.py

# │

# ├── data/

# │   ├── raw/

# │   └── processed/

# │

# ├── models/

# │   ├── best\_model.joblib

# │   ├── preprocessor.joblib

# │   ├── shap\_summary.png

# │   └── shap\_single\_customer.png

# │

# ├── src/

# │   ├── data\_preprocessing.py

# │   ├── feature\_engineering.py

# │   ├── train.py

# │   ├── evaluate.py

# │   └── explain.py

# │

# ├── requirements.txt

# ├── .gitignore

# └── README.md

# ```

# 

# > Dataset files, trained model artifacts, virtual environments, and Python cache files are excluded from GitHub where appropriate.

# 

# \---

# 

# \# 🛠️ Technologies Used

# 

# \### Programming

# 

# \* Python

# 

# \### Data Science

# 

# \* Pandas

# \* NumPy

# \* Scikit-learn

# 

# \### Machine Learning

# 

# \* Logistic Regression

# \* Random Forest

# \* XGBoost

# \* LightGBM

# 

# \### Explainable AI

# 

# \* SHAP

# 

# \### Visualization

# 

# \* Matplotlib

# 

# \### Deployment

# 

# \* Streamlit

# 

# \### Development

# 

# \* Git

# \* GitHub

# 

# \---

# 

# \# 💻 Installation

# 

# \## 1. Clone the repository

# 

# ```bash

# git clone https://github.com/SaiLokesh9828/Customer\_churn\_prediction.git

# cd Customer\_churn\_prediction

# ```

# 

# \---

# 

# \## 2. Create a virtual environment

# 

# \### Windows

# 

# ```bash

# python -m venv venv

# ```

# 

# Activate it:

# 

# ```bash

# venv\\Scripts\\activate

# ```

# 

# \---

# 

# \## 3. Install dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \---

# 

# \# ▶️ Running the Project

# 

# \## Step 1 — Prepare the data

# 

# Place the dataset at:

# 

# ```text

# data/raw/telco\_churn.csv

# ```

# 

# Then run:

# 

# ```bash

# python src/data\_preprocessing.py

# ```

# 

# \---

# 

# \## Step 2 — Train the models

# 

# ```bash

# python src/train.py

# ```

# 

# This trains and compares the available models and saves the selected model.

# 

# \---

# 

# \## Step 3 — Evaluate the model

# 

# ```bash

# python src/evaluate.py

# ```

# 

# This generates evaluation metrics and visualization artifacts.

# 

# \---

# 

# \## Step 4 — Generate SHAP explanations

# 

# ```bash

# python src/explain.py

# ```

# 

# This generates:

# 

# ```text

# models/shap\_summary.png

# models/shap\_single\_customer.png

# ```

# 

# \---

# 

# \## Step 5 — Launch the application

# 

# ```bash

# python -m streamlit run app/streamlit\_app.py

# ```

# 

# The application will be available at:

# 

# ```text

# http://localhost:8501

# ```

# 

# \---

# 

# \# 📌 Key Features

# 

# \* ✅ End-to-end ML pipeline

# \* ✅ Data preprocessing

# \* ✅ Feature engineering

# \* ✅ Multiple model comparison

# \* ✅ Hyperparameter tuning

# \* ✅ Cross-validation

# \* ✅ Churn probability prediction

# \* ✅ SHAP explainability

# \* ✅ Customer-level risk prediction

# \* ✅ Interactive Streamlit interface

# \* ✅ Reproducible project structure

# 

# \---

# 

# \# 💡 Business Use Case

# 

# A telecom company could use this system to identify customers who have a high probability of churn.

# 

# For example:

# 

# ```text

# Customer

# &#x20;  ↓

# Customer information

# &#x20;  ↓

# ML model

# &#x20;  ↓

# Churn probability

# &#x20;  ↓

# Risk classification

# &#x20;  ↓

# Retention strategy

# ```

# 

# High-risk customers could potentially be targeted with:

# 

# \* Personalized offers

# \* Customer support

# \* Service upgrades

# \* Contract incentives

# \* Retention campaigns

# 

# \---

# 

# \# ⚠️ Disclaimer

# 

# This project is intended for educational and portfolio purposes.

# 

# Predictions generated by the model are estimates and should not be treated as guaranteed outcomes or as the sole basis for business decisions.

# 

# \---

# 

# \# 👨‍💻 Author

# 

# \*\*Sai Lokesh\*\*

# 

# B.Tech Computer Science / AI-ML

# 

# Interested in:

# 

# \* Machine Learning

# \* Data Science

# \* AI Engineering

# \* Explainable AI

# \* Production ML Systems

# 

# \---

# 

# \## ⭐ If you found this project useful

# 

# Consider giving the repository a ⭐ on GitHub.



