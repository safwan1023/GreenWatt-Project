# src/save_model_bundle.py
import joblib
import pandas as pd
import os

# paths
MODEL_IN = "models/wind_power_model.pkl"
BUNDLE_OUT = "models/wind_power_model_bundle.pkl"
CLEANED = "data/cleaned_turbine_data.parquet"

# load model
model = joblib.load(MODEL_IN)

# build features list from the cleaned dataset (same preprocessing as training)
df = pd.read_parquet(CLEANED)
features = list(df.drop(columns=["Target", "timestamp", "turbine_id"]).columns)

# save bundle
joblib.dump({"model": model, "features": features}, BUNDLE_OUT)
print(f"✅ Bundle saved to: {BUNDLE_OUT}")
print("Features:", features)

