import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    score = model.score(x_test, y_test)
    return y_pred, mse, score

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_test, color="blue", label="Actual Data (y_test)", alpha=0.6)
    plt.scatter(y_test, y_pred, color="red", label="Predicted Data (y_pred)", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="green", linestyle="--", label="Identity Line")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values (y_test)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.legend()
    plt.show()

def main():
    model = joblib.load("../model/bike_prediction.pkl")
    x_test = joblib.load("../data/x_test.pkl")
    y_test = joblib.load("../data/y_test.pkl")
    
    y_pred, mse, score = evaluate_model(model, x_test, y_test)
    
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Model Score (R²): {score:.4f}")
    
    plot_actual_vs_predicted(y_test, y_pred)

if __name__ == "__main__":
    main()
