# Dataset

**Source:** IBM Telco Customer Churn dataset (publicly available on Kaggle:
`blastchar/telco-customer-churn`).

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it here as:

```
data/raw/telco_churn.csv
```

**Description:** ~7,043 customer records for a fictional telecom company, with
21 columns covering:
- Demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, payment method, billing)
- Services subscribed (phone, internet, streaming, security add-ons)
- Target: `Churn` (Yes/No)

No synthetic or fabricated data is used — the pipeline expects the real CSV
at the path above. `src/data_preprocessing.py` will raise a clear error if
the file is missing rather than silently generating fake data.
