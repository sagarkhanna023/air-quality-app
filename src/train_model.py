import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Paths
INPUT_PATH = os.path.join("..", "data", "processed", "air_quality_labeled.csv")
MODEL_PATH = os.path.join("..", "models", "pollution_classifier.pkl")

# Load and preprocess data
def load_data(path):
    df = pd.read_csv(path)
    df = df[df["pollution_level"] != "Unknown"]
    df = df.dropna(subset=["pollutant_min", "pollutant_max"])
    return df

def prepare_features(df):
    X = df[["pollutant_min", "pollutant_max", "pollutant_id"]].copy()
    
    # Encode pollutant_id
    le_pollutant = LabelEncoder()
    X["pollutant_id"] = le_pollutant.fit_transform(X["pollutant_id"])
    
    # Encode labels
    y = df["pollution_level"]
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)

    return X, y_encoded, le_pollutant, le_label

# Train, evaluate, and save
def train_model(X, y, le_label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_label.classes_))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_label.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n Cross-validation accuracy scores: {scores}")
    print(f" Mean CV Accuracy: {scores.mean():.2f}")

    return model

# Save model
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f" Model saved to: {path}")

if __name__ == "__main__":
    df = load_data(INPUT_PATH)
    X, y, le_pollutant, le_label = prepare_features(df)
    model = train_model(X, y, le_label)
    save_model(model, MODEL_PATH)

    ENCODERS_PATH = os.path.join("..", "models", "label_encoders.pkl")
    joblib.dump((le_pollutant, le_label), ENCODERS_PATH)
    print(f" Encoders saved to: {ENCODERS_PATH}")