# Project Implementation Plan: CO2 Emission Predictor

This document outlines the step-by-step implementation plan used to build the **CO2 Emission Predictor** application. Use this as a reference for understanding the project architecture, security measures, and logic flow.

---

## 1. Project Initialization & Goal Definition
**Goal**: Create a web-based application that predicts CO₂ emissions for vehicles based on specific features (Model Year, Engine Size, Cylinders, etc.) using a Machine Learning model.

*   **Structure**: The project was organized into a clear directory structure to separate concerns:
    *   `backend/`: Contains the FastAPI application and model artifacts.
    *   `frontend/`: Contains the client-side code (HTML/JS).
    *   `data/`: Stores original datasets.
    *   `requirements.txt`: Tracks Python dependencies.

---

## 2. Machine Learning Model Development
Before the backend was built, the core predictive logic was established.
*   **Training Script**: A script (`backend/train_model.py`) was likely used to train a Random Forest Regressor on the vehicle dataset.
*   **Artifacts**:
    *   **Model**: The trained model was serialized and saved as `rf_model.joblib`.
    *   **Metrics**: Model performance metrics (MAE, R2 score) were saved to `metrics.json` to allow the API to report on its own accuracy.

---

## 3. Backend Setup (FastAPI)
The backend was initialized using **FastAPI** for high performance and automatic documentation.

*   **Environment Setup**:
    *   Dependencies installed: `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `python-dotenv`.
    *   Configuration management: `python-dotenv` was used to load sensitive configuration from a `.env` file (e.g., `API_KEY`, `MODEL_PATH`).

*   **Application Skeleton (`main.py`)**:
    *   Initialized `app = FastAPI()`.
    *   Configured **Logging** to track application events and errors.

---

## 4. Core Logic Integration
The backend was connected to the Machine Learning artifacts.

*   **Startup Event**:
    *   Implemented an `@app.on_event("startup")` handler to load the model into memory *once* when the server starts, preventing reloading on every request.
    *   **Integrity Check**: Added a SHA256 hash check (`MODEL_HASH_SHA256`) to ensure the model file has not been tampered with before loading.

*   **Input Validation (Pydantic)**:
    *   Defined a `CarFeatures` Pydantic model to strictly validate incoming request data (types, ranges):
        *   `model_year`: Integer (1995-2025).
        *   `engine_size`: Float.
        *   `cylinders`, `fuel_type`, `transmission`, `vehicle_class`: Required fields.

*   **Prediction Endpoint (`POST /predict`)**:
    *   Receives JSON data matching `CarFeatures`.
    *   Converts input to a Pandas DataFrame.
    *   Runs `model.predict()`.
    *   Returns the prediction result in JSON format.

---

## 5. Security Configuration
Security was a major focus ensuring the API is production-ready.

### A. Authentication Strategy (`get_api_key`)
A dual-layer authentication system was implemented:
1.  **API Key (Primary)**:
    *   Checks for a header `x-api-key`.
    *   If it matches the server-side `API_KEY`, the request is **Authorized**.
2.  **Origin Check (Browser Fallback)**:
    *   If no API key is present (typical for frontend JS), the server checks the `Origin` header.
    *   Matches against `ALLOWED_ORIGINS` (loaded from `.env`).
    *   Allows restricted access from trusted domains (or `null`/`localhost` for development).

### B. CORS (Cross-Origin Resource Sharing)
*   Configured `CORSMiddleware` to allow requests only from specific `ALLOWED_ORIGINS`.

### C. Rate Limiting
*   Implemented a simple in-memory rate limiter (`rate_limiter` dependency).
*   Tracks request timestamps per IP address.
*   **Policy**: Limits clients to 60 requests per 60 seconds to prevent abuse.

---

## 6. Testing & Validation
*   **Unit/Integration Testing**:
    *   Created `backend/test_api.py` to verify the API externally using the `requests` library.
    *   Validates scenarios:
        *   Successful prediction (200 OK).
        *   Invalid API Key (401 Unauthorized).
        *   Bad Data types (422 Unprocessable Entity).

---

## 7. Operational Readiness
*   **Metrics Endpoint**: Added `GET /metrics` to expose model accuracy (MAE, R2) for monitoring transparency.
*   **Deployment config**:
    *   The app is set up to run via `uvicorn` with host/port configurable via environment variables.

---

## Summary of Logic Flow
1.  **Server Start** -> Load ENV -> Check Model Hash -> Load Model -> Load Metrics.
2.  **Request In** -> `rate_limiter` check -> `get_api_key` check (Header vs Origin) -> Pydantic Validation.
3.  **Processing** -> Convert attributes to DataFrame -> Predict -> Return JSON.
