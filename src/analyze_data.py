import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style
sns.set(style="whitegrid")

# Path to cleaned data (adjusted for correct relative path structure)
CLEAN_DATA_PATH = os.path.join("..", "data", "processed", "air_quality_cleaned.csv")

def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    return df

def summarize_data(df):
    print(" Data Summary:")
    print(df.info())
    print("\n Descriptive Statistics:")
    print(df.describe(include='all'))

def plot_pollution_trend(df):
    """
    Plot the average pollutant levels over time.
    """
    # We will group by pollutant_id and compute the average pollutant levels
    df_grouped = df.groupby("pollutant_id")["pollutant_avg"].mean().reset_index()

    if df_grouped.empty:
        print(" No data available to plot trends.")
        return

    plt.figure(figsize=(14, 6))
    sns.barplot(x="pollutant_id", y="pollutant_avg", data=df_grouped, hue="pollutant_id", palette="coolwarm", legend=False)
    plt.title("Average Pollutant Levels")
    plt.xlabel("Pollutant")
    plt.ylabel("Average Pollution Level (µg/m³)")
    plt.tight_layout()
    plt.show()

def plot_top_cities(df, pollutant="PM2.5"):
    """
    Plot the top 10 cities with the highest average pollutant levels for a given pollutant.
    """
    if pollutant not in df['pollutant_id'].unique():
        print(f" {pollutant} data not available in dataset.")
        return
    
    city_avg = df[df['pollutant_id'] == pollutant].groupby("city")["pollutant_avg"].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=city_avg.values, y=city_avg.index, hue=city_avg.index, palette="Reds_r", legend=False)
    plt.title(f"Top 10 Cities with Highest Average {pollutant}")
    plt.xlabel(f"Average {pollutant} Level (µg/m³)")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_clean_data(CLEAN_DATA_PATH)
    summarize_data(df)
    
    # Plot pollution trends
    plot_pollution_trend(df)
    
    # Plot top cities for each pollutant
    for pollutant in ['OZONE', 'SO2', 'PM2.5', 'NO2', 'CO', 'NH3', 'PM10']:
        plot_top_cities(df, pollutant=pollutant)
