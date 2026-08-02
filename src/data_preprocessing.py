import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load the water consumption dataset.
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Handle missing values, duplicates, and inconsistencies.
    """
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort by region and date
    df = df.sort_values(by=['region', 'date']).reset_index(drop=True)
    return df

def preprocess_data(df):
    """
    Feature engineering and preprocessing steps.
    """
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Time-series features (Lags and Rolling Averages per region)
    dfs = []
    for region, group in df.groupby('region'):
        group = group.copy()
        # Lag features
        group['lag_1'] = group['consumption_liters'].shift(1)
        group['lag_7'] = group['consumption_liters'].shift(7)
        # Rolling averages
        group['rolling_mean_7'] = group['consumption_liters'].shift(1).rolling(window=7).mean()
        group['rolling_std_7'] = group['consumption_liters'].shift(1).rolling(window=7).std()
        
        dfs.append(group)
        
    processed_df = pd.concat(dfs).reset_index(drop=True)
    
    # Drop NaNs created by lagging/rolling
    processed_df = processed_df.dropna().reset_index(drop=True)
    return processed_df

def split_data(df):
    """
    Split data temporally into train (70%), val (15%), and test (15%).
    """
    # Sort by date just to be sure
    df = df.sort_values('date').reset_index(drop=True)
    unique_dates = df['date'].unique()
    n_dates = len(unique_dates)
    
    train_idx = int(n_dates * 0.7)
    val_idx = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    
    train_df = df[df['date'].isin(train_dates)].reset_index(drop=True)
    val_df = df[df['date'].isin(val_dates)].reset_index(drop=True)
    test_df = df[df['date'].isin(test_dates)].reset_index(drop=True)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    raw_path = "Dataset/raw/water_consumption_forecasting.csv"
    processed_dir = "Dataset/processed"
    
    if os.path.exists(raw_path):
        print(f"Loading data from {raw_path}")
        df = load_data(raw_path)
        df = clean_data(df)
        df_processed = preprocess_data(df)
        
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save full processed dataset
        processed_path = os.path.join(processed_dir, "water_consumption_processed.csv")
        df_processed.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")
        
        # Split and save
        train_df, val_df, test_df = split_data(df_processed)
        train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
        print("Train, Validation, and Test datasets saved.")
    else:
        print(f"File not found: {raw_path}")
