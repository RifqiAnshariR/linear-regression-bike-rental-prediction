import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("../data/bikes.csv")

# Data Info
print("📌 First 5 Rows of Data:\n", df.head())
print("\n🔍 Data Info:")
print(df.info())
print("\n📊 Descriptive Statistics:\n", df.describe())
print("\n⚠ Missing Values Per Column:\n", df.isnull().sum())
print("\n📌 Column Names:", df.columns.tolist())

# Plot Feature vs Label
features = ["temperature", "humidity", "windspeed"]
colors = ["blue", "orange", "green"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, feature, color in zip(axes, features, colors):
    ax.scatter(df[feature], df["rentals"], color=color, alpha=0.6)
    ax.set_title(f"{feature.capitalize()} vs Rentals")
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel("Rentals")
    ax.grid(True)

plt.tight_layout()
plt.show()
