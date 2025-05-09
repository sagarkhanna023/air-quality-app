import pandas as pd
import joblib

# Load model and encoders
MODEL_PATH = "models/pollution_classifier.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"
model = joblib.load(MODEL_PATH)
le_pollutant, le_label = joblib.load(ENCODERS_PATH)

# City filter helper
def get_city_pollutants(df, city):
    """Return pollutants available for a given city."""
    return df[df["city"] == city]["pollutant_id"].unique().tolist()

# Thresholds remain unchanged
POLLUTANT_THRESHOLDS = {
    "PM2.5": [(0, 30, "Good"), (31, 60, "Satisfactory"), (61, 90, "Moderate"), (91, 120, "Poor"), (121, 250, "Very Poor"), (251, float("inf"), "Severe")],
    "PM10": [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 250, "Moderate"), (251, 350, "Poor"), (351, 430, "Very Poor"), (431, float("inf"), "Severe")],
    "NO2": [(0, 40, "Good"), (41, 80, "Satisfactory"), (81, 180, "Moderate"), (181, 280, "Poor"), (281, 400, "Very Poor"), (401, float("inf"), "Severe")],
    "SO2": [(0, 40, "Good"), (41, 80, "Satisfactory"), (81, 380, "Moderate"), (381, 800, "Poor"), (801, 1600, "Very Poor"), (1601, float("inf"), "Severe")],
    "OZONE": [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 168, "Moderate"), (169, 208, "Poor"), (209, 748, "Very Poor"), (749, float("inf"), "Severe")],
    "CO": [(0, 1, "Good"), (1.1, 2, "Satisfactory"), (2.1, 10, "Moderate"), (10.1, 17, "Poor"), (17.1, 34, "Very Poor"), (34.1, float("inf"), "Severe")],
    "NH3": [(0, 200, "Good"), (201, 400, "Satisfactory"), (401, 800, "Moderate"), (801, 1200, "Poor"), (1201, 1800, "Very Poor"), (1801, float("inf"), "Severe")]
}

# Updated predict function: no avg, includes encoded pollutant_id
def predict_aqi(pollutant_id, pollutant_min, pollutant_max):
    if not isinstance(pollutant_id, str):
        pollutant_id = str(pollutant_id)

    encoded_pollutant = le_pollutant.transform([pollutant_id])[0]

    # Match training order explicitly
    input_df = pd.DataFrame([[
        pollutant_min,
        pollutant_max,
        encoded_pollutant
    ]], columns=["pollutant_min", "pollutant_max", "pollutant_id"])

    prediction_encoded = model.predict(input_df)[0]
    return le_label.inverse_transform([prediction_encoded])[0]

# For showing thresholds in the UI
def get_threshold_table(pollutant, display_friendly=False):
    if pollutant in POLLUTANT_THRESHOLDS:
        rows = []
        for low, high, label in POLLUTANT_THRESHOLDS[pollutant]:
            if display_friendly:
                high_display = "∞" if high == float("inf") else str(high)
                low_display = str(low)
            else:
                high_display = high
                low_display = low

            rows.append({
                "Min": low_display,
                "Max": high_display,
                "Label": label
            })
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["Min", "Max", "Label"])

def get_station_pollution_summary(df):
    summaries = []
    for _, row in df.iterrows():
        city = row["city"]
        pollutant = row["pollutant_id"]
        min_val = row["pollutant_min"]
        max_val = row["pollutant_max"]
        lat = row["latitude"]
        lon = row["longitude"]
        summaries.append((city, pollutant, min_val, max_val, lat, lon))
    return summaries
