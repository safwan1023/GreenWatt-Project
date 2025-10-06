# src/fault_detection.py
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("reports", exist_ok=True)
df = pd.read_parquet("data/cleaned_turbine_data.parquet")
X = df.drop(columns=["Target","timestamp","turbine_id"])
y = df["Target"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
bundle = joblib.load("models/wind_power_model_bundle.pkl")
model = bundle["model"]
y_pred = model.predict(X_test)
res = y_test - y_pred
threshold = 3 * res.std()
anomalies = X_test[(res.abs() > threshold)].copy()
anomalies["residual"] = (res[res.abs() > threshold])
anomalies.to_csv("reports/anomalies.csv", index=False)
print(f"✅ Anomalies saved: {len(anomalies)} rows -> reports/anomalies.csv")
