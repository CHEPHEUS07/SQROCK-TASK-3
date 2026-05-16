"""
generate_data.py
================
Generates a synthetic Telco-style customer churn dataset
and saves it as 'telco_churn.csv' in the current directory.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043  # mimic real Telco dataset size

# ── Demographics ──────────────────────────────────────────
gender            = np.random.choice(["Male", "Female"], N)
senior_citizen    = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner           = np.random.choice(["Yes", "No"], N)
dependents        = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

# ── Account / contract ────────────────────────────────────
tenure            = np.random.randint(0, 73, N)
contract          = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method    = np.random.choice(
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"], N
)

# ── Services ──────────────────────────────────────────────
phone_service     = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines    = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N)
)
internet_service  = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]
)

def internet_add_on(internet_arr, yes_p=0.5):
    return np.where(
        internet_arr == "No", "No internet service",
        np.random.choice(["Yes", "No"], N, p=[yes_p, 1 - yes_p])
    )

online_security   = internet_add_on(internet_service, 0.28)
online_backup     = internet_add_on(internet_service, 0.34)
device_protection = internet_add_on(internet_service, 0.34)
tech_support      = internet_add_on(internet_service, 0.29)
streaming_tv      = internet_add_on(internet_service, 0.38)
streaming_movies  = internet_add_on(internet_service, 0.39)

# ── Charges ───────────────────────────────────────────────
monthly_charges   = np.round(
    np.random.uniform(18, 120, N), 2
)
total_charges_raw = monthly_charges * tenure + np.random.uniform(-5, 5, N)
# introduce ~11 missing total_charges (mirrors real dataset)
missing_idx = np.random.choice(N, 11, replace=False)
total_charges     = total_charges_raw.astype(object)
total_charges[missing_idx] = np.nan
total_charges     = np.round(total_charges.astype(float), 2)

# ── Churn label (probability influenced by key features) ──
churn_prob = (
    0.05
    + 0.25 * (contract == "Month-to-month")
    + 0.15 * (internet_service == "Fiber optic")
    - 0.10 * (tenure > 36)
    + 0.10 * (senior_citizen == 1)
    - 0.05 * (partner == "Yes")
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn      = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

# ── Assemble DataFrame ────────────────────────────────────
df = pd.DataFrame({
    "customerID":        [f"CUST-{i:05d}" for i in range(N)],
    "gender":            gender,
    "SeniorCitizen":     senior_citizen,
    "Partner":           partner,
    "Dependents":        dependents,
    "tenure":            tenure,
    "PhoneService":      phone_service,
    "MultipleLines":     multiple_lines,
    "InternetService":   internet_service,
    "OnlineSecurity":    online_security,
    "OnlineBackup":      online_backup,
    "DeviceProtection":  device_protection,
    "TechSupport":       tech_support,
    "StreamingTV":       streaming_tv,
    "StreamingMovies":   streaming_movies,
    "Contract":          contract,
    "PaperlessBilling":  paperless_billing,
    "PaymentMethod":     payment_method,
    "MonthlyCharges":    monthly_charges,
    "TotalCharges":      total_charges,
    "Churn":             churn,
})

df.to_csv("telco_churn.csv", index=False)
print(f"✅ Dataset generated: {N} rows × {df.shape[1]} columns")
print(f"   Churn rate: {(df['Churn']=='Yes').mean():.1%}")
print(f"   Saved → telco_churn.csv")
