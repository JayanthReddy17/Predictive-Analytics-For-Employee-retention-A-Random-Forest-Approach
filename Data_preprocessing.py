import pandas as pd

def preprocess_data(data):
    # Example: Handle missing values and encode categorical variables
    data.fillna(data.median(), inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    return data
