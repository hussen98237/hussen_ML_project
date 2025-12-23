import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
# 1. Load Data
# Define base directory (where this script is located: backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data is in ../data/RF_shuffled_data.xlsx relative to backend/
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'RF_shuffled_data.xlsx')

print(f"Loading data from {DATA_PATH}...")
df = pd.read_excel(DATA_PATH)

# 2. Define Features and Target
# Features identified: ['Model year', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Transmission', 'Vehicle class']
# Target: 'CO2 emissions (g/km)'
features = ['Model year', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Transmission', 'Vehicle class']
target = 'CO2 emissions (g/km)'

X = df[features]
y = df[target]

print("Features:", features)
print("Target:", target)

# 3. Preprocessing Pipeline
# 'Fuel type', 'Transmission', 'Vehicle class' are categorical -> OneHotEncoder
# 'Model year', 'Engine size (L)', 'Cylinders' are numeric -> Passthrough
categorical_features = ['Fuel type', 'Transmission', 'Vehicle class']
numeric_features = ['Model year', 'Engine size (L)', 'Cylinders']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 4. Create Model Pipeline
# Using RandomForestRegressor as requested
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Split Data (70% Train, 30% Test)
print("Splitting data 70/30...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Train Model
print("Training model...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluate
print("Evaluating model...")
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Save Model
model_filename = os.path.join(BASE_DIR, 'rf_model.joblib')
joblib.dump(model_pipeline, model_filename)
print(f"Model saved to {model_filename}")

# 9. Save Metrics (Optional, for frontend/backend usage)
metrics = {
    "mae": round(mae, 2),
    "r2": round(r2, 4)
}
metrics_path = os.path.join(BASE_DIR, 'metrics.json')
with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print("Metrics saved to metrics.json")
