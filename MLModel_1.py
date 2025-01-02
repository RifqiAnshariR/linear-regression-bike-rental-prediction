import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, linear_model, metrics

# Load data
bikes = pd.read_csv('bikes.csv')

# Plot for each feature vs rentals
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(bikes['temperature'], bikes['rentals'], color='blue', alpha=0.6)
axes[0].set_title('Temperature vs Rentals')
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('Rentals')

axes[1].scatter(bikes['humidity'], bikes['rentals'], color='orange', alpha=0.6)
axes[1].set_title('Humidity vs Rentals')
axes[1].set_xlabel('Humidity')
axes[1].set_ylabel('Rentals')

axes[2].scatter(bikes['windspeed'], bikes['rentals'], color='green', alpha=0.6)
axes[2].set_title('Windspeed vs Rentals')
axes[2].set_xlabel('Windspeed')
axes[2].set_ylabel('Rentals')

plt.tight_layout()  # Adjust layout to prevent overlapping

# Prepare data
response = 'rentals'
y = bikes[[response]]  # Target variable (rentals)

# Features (predictors)
predictors = [col for col in bikes.columns if col != response]
x = bikes[predictors]  # Feature variables (temperature, humidity, windspeed)

# Train-Test Split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1234)

# Train Linear Regression Model
model = linear_model.LinearRegression().fit(x_train, y_train)

# Predict on Test Data
y_pred = model.predict(x_test)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))

plt.scatter(y_test, y_test, color="blue", label="Actual Data (y_test)", alpha=0.6)
plt.scatter(y_test, y_pred, color="red", label="Predicted Data (y_pred)", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="green", linestyle="--", label="Identity Line") # Identity Line

# Plot settings
plt.title("Actual vs Predicted")
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.legend()

plt.show()

# Model Coefficients and Intercept
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Model Score:", model.score(x_test, y_test))