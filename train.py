import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATASET_PATH = "dataset/winequality-white.csv"
OUTPUT_DIR = "outputs"

print("Loading dataset from:", DATASET_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATASET_PATH, sep=";")

print("Dataset shape:", data.shape)
print("Columns:", list(data.columns))

X = data.drop("quality", axis=1)
y = data["quality"]

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed")

y_pred = model.predict(X_test)

print("Prediction completed")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

model_path = os.path.join(OUTPUT_DIR, "model.pkl")
joblib.dump(model, model_path)

print("Model saved to:", model_path)

results = {
    "MSE": mse,
    "R2_Score": r2
}

results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to:", results_path)
print("Training pipeline completed successfully")
