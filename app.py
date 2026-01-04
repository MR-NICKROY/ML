import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI

app = FastAPI()

# ================= LOAD MODEL FILES =================
model = joblib.load("fraud_model.pkl")
encoder = joblib.load("category_encoder.pkl")
model_features = joblib.load("model_features.pkl")


@app.get("/")
def health():
    return {"status": "ML service running"}


@app.post("/predict")
def predict(input_data: dict):

    df = pd.DataFrame([input_data])

    # ---------- Feature Engineering ----------
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")

    df["gap"] = (df["Timestamp"] - df["LastLogin"]).dt.days.abs().fillna(0)
    df["Hour"] = df["Timestamp"].dt.hour.fillna(0)

    def first_digit(x):
        s = str(abs(x)).replace(".", "").lstrip("0")
        return int(s[0]) if s else 0

    df["Benford_Prob"] = df["TransactionAmount"].apply(first_digit).map(
        {d: np.log10(1 + 1/d) for d in range(1, 10)}
    ).fillna(0)

    df["Transaction_Frequency"] = input_data.get("Transaction_Frequency", 1)
    df["Total_Linked_Value"] = input_data.get(
        "Total_Linked_Value", df["TransactionAmount"]
    )
    df["SuspiciousFlag"] = input_data.get("SuspiciousFlag", 0)

    try:
        df["Category"] = encoder.transform(df["Category"])
    except:
        df["Category"] = -1

    # ---------- Align Features ----------
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df_final = df[model_features]

    # ---------- Prediction ----------
    pred = model.predict(df_final)[0]
    prob = model.predict_proba(df_final)[0][1]

    return {
        "is_fraud": bool(pred),
        "risk_score": float(prob)
    }
