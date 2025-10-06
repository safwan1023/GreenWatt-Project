import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Step 1: Load cleaned dataset
df = pd.read_parquet("data/cleaned_turbine_data.parquet")
print("✅ Cleaned dataset loaded!")

# Step 2: Separate features (X) and target (y)
X = df.drop(columns=["Target", "timestamp", "turbine_id"])
y = df["Target"]

# Step 3: Take a smaller sample (optional but faster)
df_sample = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
X = df_sample.drop(columns=["Target", "timestamp", "turbine_id"])
y = df_sample["Target"]

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train a smaller Random Forest (faster)
model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
print("⚙️ Training model... please wait a few seconds...")
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Save model
joblib.dump(model, "models/wind_power_model.pkl")
print("💾 Model saved to models/wind_power_model.pkl")
