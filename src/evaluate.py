# src/evaluate.py
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure directories exist
os.makedirs("reports/figures", exist_ok=True)

# Load cleaned data
df = pd.read_parquet("data/cleaned_turbine_data.parquet")
X = df.drop(columns=["Target", "timestamp", "turbine_id"])
y = df["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model bundle (fallback to plain model)
bundle_path = "models/wind_power_model_bundle.pkl"
if os.path.exists(bundle_path):
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
else:
    model = joblib.load("models/wind_power_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Fixed for older sklearn versions
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Save metrics to text
with open("reports/metrics.txt", "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")

# Parity plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=4, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Parity plot")
plt.savefig("reports/figures/parity.png", bbox_inches="tight")
plt.clf()

# Residuals histogram
res = y_test - y_pred
plt.hist(res, bins=50)
plt.xlabel("Residual (Actual - Predicted)")
plt.title("Residual histogram")
plt.savefig("reports/figures/residuals.png", bbox_inches="tight")
plt.clf()

print("✅ Evaluation done. Plots saved to reports/figures/ and metrics to reports/metrics.txt")
