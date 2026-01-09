import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATASET_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATASET_PATH, sep=";")

selected_features = ["alcohol", "sulphates", "volatile acidity"]
X = data[selected_features]
y = data["quality"]



MODEL_TYPE = "rf"
USE_SCALER = False
TEST_SIZE = 0.3




if USE_SCALER:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(
        {
            "model": MODEL_TYPE,
            "scaler": USE_SCALER,
            "test_size": TEST_SIZE,
            "mse": mse,
            "r2": r2
        },
        f,
        indent=4
    )
