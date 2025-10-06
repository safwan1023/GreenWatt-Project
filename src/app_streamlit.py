# src/app_streamlit.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="GreenWatt - Turbine Power Predictor", layout="wide")

# load model bundle
bundle_path = "models/wind_power_model_bundle.pkl"
if not os.path.exists(bundle_path):
    st.error("Model bundle not found. Run src/save_model_bundle.py first.")
    st.stop()

bundle = joblib.load(bundle_path)
model = bundle["model"]
features = bundle["features"]

st.title("GreenWatt — Turbine Power Prediction")
st.write("Enter sensor values (numeric). Leave defaults if unknown.")

# build inputs
inputs = {}
cols = st.columns(2)
for i, f in enumerate(features):
    default = 0.0
    inputs[f] = cols[i % 2].number_input(label=f, value=float(default), format="%.4f")

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=features)
    pred = model.predict(X)[0]
    st.metric("Predicted Power (kW)", f"{pred:.2f}")
    # show top feature importances
    fi = model.feature_importances_
    fi_df = pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False).reset_index(drop=True)
    st.subheader("Top feature importances")
    st.table(fi_df.head(10))

st.markdown("---")
st.write("Model info:")
st.write(f"Model type: {type(model).__name__}")
