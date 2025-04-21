import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Sample input
input_data = pd.read_csv("data/sample_input.csv")

# Make predictions
predictions = model.predict(input_data)
print("Predictions:", predictions)
