# src/explainability_shap.py
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

os.makedirs("reports/figures", exist_ok=True)

# load model bundle
bundle_path = "models/wind_power_model_bundle.pkl"
if not os.path.exists(bundle_path):
    raise FileNotFoundError(f"Bundle not found: {bundle_path}. Run src/save_model_bundle.py first.")

bundle = joblib.load(bundle_path)
model = bundle["model"]
features = bundle["features"]

# load data & sample
df = pd.read_parquet("data/cleaned_turbine_data.parquet")
X = df[features]
X_sample = X.sample(n=200, random_state=42)

# SHAP (TreeExplainer for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# summary plot (be sure matplotlib backend used above)
shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight")
plt.clf()

print("✅ SHAP summary saved to reports/figures/shap_summary.png")
