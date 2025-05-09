# Save both encoders for prediction
joblib.dump((le_pollutant, le_label), os.path.join("..", "models", "label_encoders.pkl"))
