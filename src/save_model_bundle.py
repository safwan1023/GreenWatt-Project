# src/save_model_bundle.py
import joblib
import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script is
MODEL_IN = os.path.join(BASE_DIR, "../models/wind_power_model.pkl")
BUNDLE_OUT = os.path.join(BASE_DIR, "../models/wind_power_model_bundle.pkl")
CLEANED = os.path.join(BASE_DIR, "../data/cleaned_turbine_data.parquet")

# --- Load model ---
if not os.path.exists(MODEL_IN):
    raise FileNotFoundError(f"Model file not found: {MODEL_IN}")
model = joblib.load(MODEL_IN)

# --- Build features list from the cleaned dataset ---
if not os.path.exists(CLEANED):
    raise FileNotFoundError(f"Cleaned data file not found: {CLEANED}")
df = pd.read_parquet(CLEANED)
features = list(df.drop(columns=["Target", "timestamp", "turbine_id"]).columns)

# --- Save bundle ---
os.makedirs(os.path.dirname(BUNDLE_OUT), exist_ok=True)
joblib.dump({"model": model, "features": features}, BUNDLE_OUT)

print(f"✅ Bundle saved to: {BUNDLE_OUT}")
print("Features:", features)
