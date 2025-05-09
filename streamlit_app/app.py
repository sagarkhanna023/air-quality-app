# streamlit_app/app.py

import streamlit as st
import pandas as pd
import joblib
import os
from utils import classify_aqi_from_thresholds, predict_aqi, get_threshold_table, get_city_pollutants, get_station_pollution_summary 
import folium
from streamlit_folium import st_folium
import plotly.express as px
import base64

# Load model and data
MODEL_PATH = "models/pollution_classifier.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
DATA_PATH = "data/processed/air_quality_cleaned.csv"

# Load the model, label encoders, and dataset
model = joblib.load(MODEL_PATH)
le_pollutant, le_label = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

st.set_page_config(layout="wide", page_title="Air Quality Predictor 🇮🇳")
st.title("Indian City Air Quality Predictor")

st.markdown("""
This app predicts **AQI Level** based on pollutant readings.

Select a pollutant, enter minimum and maximum observed values, and we’ll predict the air quality category.
""")

# --- Main Pollutant Input & Prediction ---
pollutant = st.selectbox("Select Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "NH3"])

# Show classification thresholds
st.markdown("####  AQI Classification Table")
st.dataframe(get_threshold_table(pollutant, display_friendly=True), use_container_width=True)

# Inputs for prediction
col1, col2 = st.columns(2)
with col1:
    pollutant_min = st.number_input(f"Enter Minimum {pollutant} Value", min_value=0.0)
with col2:
    pollutant_max = st.number_input(f"Enter Maximum {pollutant} Value", min_value=0.0)

method = st.radio("Select Prediction Method", ["Model-Based", "Threshold-Based"], horizontal=True)

if st.button("Predict AQI Level"):
    if method == "Model-Based":
        prediction = predict_aqi(pollutant, pollutant_min, pollutant_max)
        st.success(f"🔍 **Model Prediction**: {prediction}")
    else:
        prediction = classify_aqi_from_thresholds(pollutant, pollutant_min, pollutant_max)
        st.success(f"📏 **Threshold-Based Classification**: {prediction}")
# --- Sidebar: City-based selection ---
st.sidebar.header("Choose from City Data")

city = st.sidebar.selectbox("City", sorted(df["city"].unique()))
available_pollutants = get_city_pollutants(df, city)
pollutant_sidebar = st.sidebar.selectbox("Pollutant", sorted(available_pollutants))

# Get min/max from data for context (optional default values)
city_pollutant_data = df[(df["city"] == city) & (df["pollutant_id"] == pollutant_sidebar)]
default_min = city_pollutant_data["pollutant_min"].mean()
default_max = city_pollutant_data["pollutant_max"].mean()

min_val = st.sidebar.number_input(f"Min {pollutant_sidebar}", min_value=0.0, value=float(default_min or 0.0))
max_val = st.sidebar.number_input(f"Max {pollutant_sidebar}", min_value=0.0, value=float(default_max or 0.0))

if st.sidebar.button("Predict City Pollution Level"):
    prediction = predict_aqi(pollutant_sidebar, min_val, max_val)
    st.sidebar.success(f"Predicted Air Quality Level: **{prediction}**")

# ---------- AQI Level Distribution Chart ----------
st.markdown("## AQI Level Distribution by Station")

aqi_df = df.copy()
aqi_df["AQI_Level"] = aqi_df.apply(
    lambda row: predict_aqi(row["pollutant_id"], row["pollutant_min"], row["pollutant_max"]),
    axis=1
)

fig = px.histogram(
    aqi_df,
    x="AQI_Level",
    color="AQI_Level",
    title="Air Quality Category Distribution",
    category_orders={"AQI_Level": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]},
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Download Button ----------
st.markdown("## Download Filtered City Data")
filtered_df = df[df["city"] == city]
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download City Data as CSV",
    data=csv,
    file_name=f"{city}_air_quality.csv",
    mime='text/csv'
)

# ---------- Quick Insights ----------
st.markdown("## 🔍 Quick Insights")

worst_city = (
    aqi_df.groupby("city")["AQI_Level"]
    .apply(lambda x: (x == "Severe").sum())
    .sort_values(ascending=False)
    .idxmax()
)

best_city = (
    aqi_df.groupby("city")["AQI_Level"]
    .apply(lambda x: (x == "Good").sum())
    .sort_values(ascending=False)
    .idxmax()
)

common_pollutant = df["pollutant_id"].mode()[0]

st.write(f"🏭 **City with most 'Severe' AQI days:** {worst_city}")
st.write(f"🌳 **City with most 'Good' AQI days:** {best_city}")
st.write(f"🧪 **Most common pollutant recorded:** {common_pollutant}")    

# --- Data Preview ---
with st.expander(" Raw Data Preview"):
    st.dataframe(df[df["city"] == city])

st.markdown("##  City-wise Air Quality Map")

summary = get_station_pollution_summary(df)

m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

for city, pollutant, min_val, max_val, lat, lon in summary:
    aqi_level = predict_aqi(pollutant, min_val, max_val)


    color = {
        "Good": "green",
        "Satisfactory": "lightgreen",
        "Moderate": "orange",
        "Poor": "red",
        "Very Poor": "darkred",
        "Severe": "black"
    }.get(aqi_level, "gray")

    folium.CircleMarker(
        location=(lat, lon),
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"{city} ({pollutant}): {aqi_level}"
    ).add_to(m)

st.markdown("##  Station-wise Air Quality Map")
st_folium(m, width=1200, height=600)

