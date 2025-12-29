from fastapi import FastAPI, HTTPException, Security, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import os
import hashlib
import logging
import time
from collections import defaultdict
from dotenv import load_dotenv

# Load Environment Variables from .env file in the same directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize App
app = FastAPI(title="CO2 Emission Predictor API")

# 1. Configuration & Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, "rf_model.joblib"))
METRICS_PATH = os.getenv('METRICS_PATH', os.path.join(BASE_DIR, "metrics.json"))
ALLOWED_ORIGINS = json.loads(os.getenv('ALLOWED_ORIGINS', '["*"]'))
API_KEY = os.getenv('API_KEY', 'change_me_to_a_secure_key') # Default for dev, warn in prod
MODEL_HASH_SHA256 = os.getenv('MODEL_HASH_SHA256')

# 2. CORS
# CORS is intentionally permissive for development and portfolio/demo purposes.
# In production, allowed origins MUST be restricted to trusted frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
metrics_data = {}

# Rate Limiting (Simple In-Memory), NOT adequate for production. Suggest Redis or similar for distributed rate limiting.
# Map IP -> list of timestamps
rate_limit_store = defaultdict(list)
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 60  # requests per duration

# 3. Security Utilities
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

async def get_api_key(
    api_key_header: str = Security(api_key_header),
    request: Request = None
):
    # lines from 72 to 79 works when you run test_api.py, but when you run prediction from frontend it doesn't work
    if api_key_header:
        #1. Check if API Key is valid (Highest Priority - for Scripts/Server-to-Server)
        if api_key_header != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key, please check your API key and try again.",
            )
    # 2. Origin Check (Fallback - for Trusted Frontend)
    # Browsers send 'Origin' header. Non-browsers can spoof it, but we combine this 
    # with Rate Limiting to protect the public interface.
    if request: 
        origin = request.headers.get("origin")
        if origin and origin in ALLOWED_ORIGINS:
            return "authorized_by_origin"
        
        # Allow requests from localhost/null (often used during local dev/double-click)
        # IF configured to do so.
        if origin is None and "null" in ALLOWED_ORIGINS:
             return "authorized_by_null_origin"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key or Unauthorized Origin",
    )

async def rate_limiter(request: Request):
    client_ip = request.client.host
    now = time.time()
    
    # Filter out timestamps older than the duration
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_LIMIT_DURATION]
    
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    rate_limit_store[client_ip].append(now)

# 4. Startup Event
@app.on_event("startup")
def load_artifacts():
    global model, metrics_data
    try:
        # Load Model with Integrity Check
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        if not os.path.exists(MODEL_PATH):
             logger.error(f"Model file not found at {MODEL_PATH}")
             return

        # Integrity Check
        if MODEL_HASH_SHA256:
            current_hash = calculate_file_hash(MODEL_PATH)
            if current_hash != MODEL_HASH_SHA256:
                logger.critical(f"Security Alert: Model hash mismatch! Expected {MODEL_HASH_SHA256}, got {current_hash}")
                # We do NOT load the model if hash mismatches
                return
            logger.info("Model integrity verified.")
        else:
            logger.warning("No MODEL_HASH_SHA256 provided in environment. Skipping integrity check.")

        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        
        # Load Metrics
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
            logger.info("Metrics loaded.")
        else:
            logger.warning("Metrics file not found. Using defaults.")
            metrics_data = {"mae": "N/A", "r2": "N/A"}
            
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        # Fail safe: Ensure model is None if loading fails

# Input Schema
class CarFeatures(BaseModel):
    model_year: int = Field(..., ge=1995, le=2025, description="Model year between 1995 and 2025")
    engine_size: float
    cylinders: int
    fuel_type: str
    transmission: str
    vehicle_class: str

# Endpoints
@app.get("/", dependencies=[Depends(rate_limiter)])
def read_root():
    return {"message": "CO2 Prediction API is running"}

@app.get("/metrics", dependencies=[Depends(rate_limiter)])
def get_metrics():
    """Returns model evaluation metrics."""
    return {
        "model_name": "RandomForestRegressor",
        "mae": metrics_data.get("mae"),
        "r2": metrics_data.get("r2")
    }

@app.post("/predict", dependencies=[Depends(get_api_key), Depends(rate_limiter)])
def predict_emission(features: CarFeatures):
    """Predicts CO2 emission based on car features."""
    if model is None:
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not initialized.")
    
    try:
        # Prepare data for prediction
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
        # Log the full error but return generic to client
        # Full exception details are logged for debugging purposes only.
        # Logs must be properly secured and never exposed to end users or public systems,
        # as they may contain sensitive internal information.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # For external/production deployments, further web server security and firewalls 
    # (or running on localhost) are recommended.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
