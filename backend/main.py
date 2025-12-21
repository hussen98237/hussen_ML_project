from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os

# Initialize App
app = FastAPI(title="CO2 Emission Predictor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In prod, specify frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Global variables for model and metrics
model = None
metrics_data = {}

# Startup Event
@app.on_event("startup")
def load_artifacts():
    global model, metrics_data
    try:
        # Load Model
        print(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        
        # Load Metrics
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
            print("Metrics loaded.")
        else:
            print("Metrics file not found. Using defaults.")
            metrics_data = {"mae": "N/A", "r2": "N/A"}
            
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # In a real app, we might want to crash if model fails, but for dev we'll log it.

# Input Schema
from pydantic import BaseModel, Field

# ... (rest of imports)

# Input Schema
class CarFeatures(BaseModel):
    model_year: int = Field(..., ge=1995, le=2025, description="Model year between 1995 and 2025")
    engine_size: float
    cylinders: int
    fuel_type: str
    transmission: str
    vehicle_class: str

# Endpoints
@app.get("/")
def read_root():
    return {"message": "CO2 Prediction API is running"}

@app.get("/metrics")
def get_metrics():
    """Returns model evaluation metrics."""
    return {
        "model_name": "RandomForestRegressor",
        "mae": metrics_data.get("mae"),
        "r2": metrics_data.get("r2")
    }

@app.post("/predict")
def predict_emission(features: CarFeatures):
    """Predicts CO2 emission based on car features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Prepare data for prediction
        # Map Pydantic fields to DataFrame columns expected by the model pipeline
        input_data = {
            "Model year": [features.model_year],
            "Engine size (L)": [features.engine_size],
            "Cylinders": [features.cylinders],
            "Fuel type": [features.fuel_type],
            "Transmission": [features.transmission],
            "Vehicle class": [features.vehicle_class]
        }
        
        df = pd.DataFrame(input_data)
        
        # Prediction
        prediction = model.predict(df)[0]
        
        return {
            "prediction": float(prediction)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# For debugging directly with python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
