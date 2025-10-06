import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/turbine_data.csv")

# Step 2: Optional - show first 5 rows
print("✅ Dataset loaded successfully!")
print(df.head())

# Step 3: Data cleaning (basic example)
df = df.dropna()  # remove missing rows

# Step 4: Save cleaned data
df.to_parquet("data/cleaned_turbine_data.parquet", index=False)

print("✅ Cleaned data saved to data/cleaned_turbine_data.parquet")
