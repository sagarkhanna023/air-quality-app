# src/clean_data.py

import pandas as pd
import os

# Define file paths
RAW_DATA_PATH = os.path.join("..", "data", "raw", "air_quality_india.csv")
PROCESSED_DATA_PATH = os.path.join("..", "data", "processed", "air_quality_cleaned.csv")

def load_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Drop rows missing essential info (excluding last_update now)
    df = df.dropna(subset=[
        "pollutant_id", "pollutant_avg",
        "country", "state", "city", "station", "latitude", "longitude"
    ])
    
    # Drop 'last_update' column completely
    if 'last_update' in df.columns:
        df.drop(columns=["last_update"], inplace=True)

    # Fill missing pollutant values with column-wise mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Round all float columns to 2 decimal places
    df = df.round(2)

    return df

def save_clean_data(df, output_path):
    print(f"Saving cleaned data to: {output_path}")
    
    # Save cleaned data to CSV without the index
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Step 1: Load raw data
    df = load_data(RAW_DATA_PATH)

    # Step 2: Clean the data (no pivoting, just keep pollutant_id)
    df_clean = clean_data(df)

    print(df_clean.head())

    # Step 3: Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Step 4: Save cleaned data
    save_clean_data(df_clean, PROCESSED_DATA_PATH)

    print("Data cleaning complete.")
