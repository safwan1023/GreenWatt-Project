import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'wind_speed' in df.columns:
        df['wind_speed_sq'] = df['wind_speed'] ** 2
        df['wind_speed_cu'] = df['wind_speed'] ** 3

    if 'timestamp' in df.columns:
        df['ws_roll_5'] = df['wind_speed'].rolling(window=5, min_periods=1).mean()
        df['power_roll_5'] = df['power'].rolling(window=5, min_periods=1).mean()

    if 'rated_power' in df.columns:
        df['capacity_frac'] = df['power'] / df['rated_power']
    return df

def prepare_features(df: pd.DataFrame, scaler: StandardScaler = None):
    features = [c for c in df.columns if df[c].dtype in ['float64','int64'] and c != 'power']
    X = df[features].fillna(0)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, df['power'].values, features, scaler
