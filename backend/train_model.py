import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os
import hashlib
import logging
from dotenv import load_dotenv

# Load Environment Variables from .env file in the same directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # 1. Load Configuration
    # Define base directory (where this script is located: backend/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load paths from env or use defaults (relative to BASE_DIR)
    DATA_PATH = os.getenv('DATA_PATH', os.path.join(BASE_DIR, '..', 'data', 'RF_shuffled_data.xlsx'))
    MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'rf_model.joblib'))
    METRICS_PATH = os.getenv('METRICS_PATH', os.path.join(BASE_DIR, 'metrics.json'))

    # Load Data
    logger.info(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Data file not found at {DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Define Features and Target
    features = ['Model year', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Transmission', 'Vehicle class']
    target = 'CO2 emissions (g/km)'

    if not all(col in df.columns for col in features + [target]):
        logger.error("Missing required columns in dataset.")
        return

    X = df[features]
    y = df[target]

    logger.info(f"Features: {features}")
    logger.info(f"Target: {target}")

    # 3. Preprocessing Pipeline
    categorical_features = ['Fuel type', 'Transmission', 'Vehicle class']
    numeric_features = ['Model year', 'Engine size (L)', 'Cylinders']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # 4. Create Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 5. Split Data (70% Train, 30% Test)
    logger.info("Splitting data 70/30...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 6. Train Model
    logger.info("Training model...")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluate
    logger.info("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")

    # 8. Save Model
    try:
        joblib.dump(model_pipeline, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Calculate and log the hash for environment configuration
        model_hash = calculate_file_hash(MODEL_PATH)
        logger.info(f"Model SHA256 Hash: {model_hash}")
        logger.info("IMPORTANT: Update MODEL_HASH_SHA256 in your .env file with this value.")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # 9. Save Metrics
    metrics = {
        "mae": round(mae, 2),
        "r2": round(r2, 4)
    }
    
    try:
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Metrics saved to {METRICS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

if __name__ == "__main__":
    main()
