import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your Node.js backend can access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL FILES =================
try:
    model = joblib.load("fraud_model.pkl")
    encoder = joblib.load("category_encoder.pkl")
    model_features = joblib.load("model_features.pkl")
    print("✅ Model files loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model files: {e}")

@app.get("/")
def health():
    return {"status": "ML service running"}

@app.post("/predict")
def predict(input_data: dict):
    # 1. ROBUST INPUT FETCHING
    sus_flag = (
        input_data.get("SuspiciousFlag") or 
        input_data.get("suspiciousFlag") or 
        input_data.get("suspicious_flag") or 
        0
    )

    clean_data = {
        "TransactionAmount": float(input_data.get("TransactionAmount", 0)),
        "AccountBalance": float(input_data.get("AccountBalance", 0)),
        "AnomalyScore": float(input_data.get("AnomalyScore", 0)),
        "Transaction_Frequency": float(input_data.get("Transaction_Frequency", 1)),
        "Total_Linked_Value": float(input_data.get("Total_Linked_Value", 0)),
        "SuspiciousFlag": int(sus_flag),
        "Timestamp": str(input_data.get("Timestamp", "")),
        "LastLogin": str(input_data.get("LastLogin", "")),
        "Category": str(input_data.get("Category", "Other")),
        "MerchantID": str(input_data.get("MerchantID", "")),
        "CustomerID": str(input_data.get("CustomerID", ""))
    }

    df = pd.DataFrame([clean_data])

    # 2. FIX DATE PARSING
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df["LastLogin"] = pd.to_datetime(df["LastLogin"], dayfirst=True, errors="coerce")
    
    # Fallback for broken dates
    if df['Timestamp'].isna().any(): df['Timestamp'] = pd.Timestamp.now()
    if df['LastLogin'].isna().any(): df['LastLogin'] = pd.Timestamp.now()

    # 3. CALCULATE FEATURES
    df["gap"] = (df["Timestamp"] - df["LastLogin"]).dt.days.abs()
    df["Hour"] = df["Timestamp"].dt.hour

    # --- HYBRID RULES LOGIC ---
    
    # A. Weekend Heist
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    is_weekend = 1 if df['DayOfWeek'][0] >= 5 else 0
    
    # B. Night Shift (1 AM - 5 AM)
    is_night = 1 if 1 <= df['Hour'][0] <= 5 else 0
    
    # C. High Value Spike (> 50,000)
    safe_avg = 10000
    is_spike = 1 if df['TransactionAmount'][0] > (5 * safe_avg) else 0

    # D. Benford's Law
    def first_digit(x):
        s = str(abs(x)).replace(".", "").lstrip("0")
        return int(s[0]) if s else 0
    df["Benford_Prob"] = df["TransactionAmount"].apply(first_digit).map(
        {d: np.log10(1 + 1/d) for d in range(1, 10)}
    ).fillna(0)

    # 4. ENCODE & ALIGN FOR MODEL
    try:
        df["Category"] = encoder.transform(df["Category"])
    except:
        df["Category"] = 0 

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df_final = df[model_features]

    # 5. PREDICT (Base ML Score)
    try:
        prob = model.predict_proba(df_final)[0][1]
    except:
        prob = 0.0

    # 6. OVERRIDE LOGIC (Force High Risk)
    if is_spike:
        prob = max(prob, 0.95)
    if is_night:
        prob = max(prob, 0.75)
    if is_weekend:
        prob = max(prob, prob + 0.1)
    if clean_data["SuspiciousFlag"] == 1:
        prob = max(prob, 0.85)

    prob = min(prob, 1.0)
    
    # 7. CONVERT TO PYTHON BOOL (Prevents Crash)
    final_is_fraud = bool(prob > 0.5)

    return {
        "is_fraud": final_is_fraud,
        "risk_score": float(prob),
        "debug_tags": {
            "is_weekend": bool(is_weekend),
            "is_night": bool(is_night),
            "is_spike": bool(is_spike),
            "original_model_score_was_low": bool(prob > 0.5)
        }
    }