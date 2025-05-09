import pandas as pd
import os

RAW_DATA_PATH = os.path.join("..", "data", "processed", "air_quality_cleaned.csv")
OUTPUT_PATH = os.path.join("..", "data", "processed", "air_quality_labeled.csv")

# Classification rules for each pollutant based on Indian CPCB AQI
POLLUTANT_THRESHOLDS = {
    "PM2.5":   [(0, 30, "Good"), (31, 60, "Satisfactory"), (61, 90, "Moderate"), (91, 120, "Poor"), (121, 250, "Very Poor"), (251, float("inf"), "Severe")],
    "PM10":    [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 250, "Moderate"), (251, 350, "Poor"), (351, 430, "Very Poor"), (431, float("inf"), "Severe")],
    "NO2":     [(0, 40, "Good"), (41, 80, "Satisfactory"), (81, 180, "Moderate"), (181, 280, "Poor"), (281, 400, "Very Poor"), (401, float("inf"), "Severe")],
    "SO2":     [(0, 40, "Good"), (41, 80, "Satisfactory"), (81, 380, "Moderate"), (381, 800, "Poor"), (801, 1600, "Very Poor"), (1601, float("inf"), "Severe")],
    "OZONE":   [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 168, "Moderate"), (169, 208, "Poor"), (209, 748, "Very Poor"), (749, float("inf"), "Severe")],
    "CO":      [(0, 1, "Good"), (1.1, 2, "Satisfactory"), (2.1, 10, "Moderate"), (10.1, 17, "Poor"), (17.1, 34, "Very Poor"), (34.1, float("inf"), "Severe")],
    "NH3":     [(0, 200, "Good"), (201, 400, "Satisfactory"), (401, 800, "Moderate"), (801, 1200, "Poor"), (1201, 1800, "Very Poor"), (1801, float("inf"), "Severe")],
}

def classify_pollutant(pollutant, value):
    if pd.isna(value) or pollutant not in POLLUTANT_THRESHOLDS:
        return "Unknown"
    
    for low, high, label in POLLUTANT_THRESHOLDS[pollutant]:
        if low <= value <= high:
            return label
    return "Unknown"

def load_data(filepath):
    return pd.read_csv(filepath)

def create_labels(df):
    df["pollution_level"] = df.apply(lambda row: classify_pollutant(row["pollutant_id"], row["pollutant_avg"]), axis=1)
    return df

def save_labeled_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f" Labeled data saved to: {output_path}")

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df_labeled = create_labels(df)
    save_labeled_data(df_labeled, OUTPUT_PATH)
