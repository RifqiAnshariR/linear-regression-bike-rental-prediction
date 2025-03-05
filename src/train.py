import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load Data
def load_data(filepath):
    return pd.read_csv(filepath)

# Prepare Data
def prepare_data(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Train Model
def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

# Run Pipeline
def main():
    # Load & prepare data
    df = load_data("../data/bikes.csv")
    x_train, x_test, y_train, y_test = prepare_data(df, target_column="rentals")

    # Train model
    model = train_model(x_train, y_train)

    # Export model & test data
    joblib.dump(model, "../model/bike_prediction.pkl")
    joblib.dump(x_test, "../data/x_test.pkl")
    joblib.dump(y_test, "../data/y_test.pkl")

if __name__ == "__main__":
    main()
