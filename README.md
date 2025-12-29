
# hussen_ML_project

# Car CO2 Emission Predictor

> [!IMPORTANT]
> **Educational Project**: This application is designed for **demonstration and training purposes only**. While it implements security best practices (like key hashing and environment-based configuration), it is not intended for mission-critical production deployment without further hardening.

This project is a Machine Learning web application designed to predict CO2 emissions from vehicles based on their specifications. It utilizes a Random Forest Regression model trained on vehicle data to provide accurate emission estimates.
## 🚀 Project Overview
The **Car CO2 Emission Predictor** solves the problem of estimating environmental impact by analyzing key vehicle characteristics. It provides a simple, user-friendly interface for users to input car details and receive instant predictions.
### Technologies Used
*   **Machine Learning**: Python, Scikit-Learn, Pandas, Joblib
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Data Processing**: Pandas, OpenPyXL
## 🛠️ Prerequisites
Before you begin, ensure you have the following installed on your system:
*   **Python 3.8+**
*   **pip** (Python package installer)
## 📦 Installation & Setup

1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <repository-url>
    cd web_app_CO2_emission_from_cars
    ```

2.  **Setup Environment Variables**:
    *   Duplicate the example configuration file:
        ```bash
        cp backend/.env.example backend/.env
        ```
    
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Training the Model & Security Setup
This project employs **integrity checks** for the Machine Learning model. You **must** generate the model and its hash before running the server.

1.  Ensure your dataset is located at `data/RF_shuffled_data.xlsx`.
2.  Run the training script:
    ```bash
    python backend/train_model.py
    ```
3.  **CRITICAL STEP**:
    *   The script will output a **SHA256 Hash** in the terminal (e.g., `Model SHA256 Hash: c9f3c8...`).
    *   Copy this hash.
    *   Open `backend/.env` and paste it into the `MODEL_HASH_SHA256` variable.
    *   *If this step is skipped, the backend will warn you and continue.*

## 🏃‍♂️ How to Run the Application

### 1. Start the Backend API
Navigate to the project root and run:
```bash
python backend/main.py
```
*   The API will start at `http://localhost:8000`.
*   You will see logs confirming configuration loading and model integrity verification.

### 2. Launch the Frontend
*   Open the `frontend` folder.
*   Double-click `index.html` to open it in your web browser.
*   **Note**: The frontend has been configured to use the API Key for demonstration. In a real production environment, you would not hardcode keys in client-side JavaScript.
## 📂 Project Structure
```
├── backend/
│   ├── main.py             # FastAPI backend application
│   ├── train_model.py      # ML model training script
│   ├── rf_model.joblib     # Saved Random Forest model (generated)
│   └── metrics.json        # Model performance metrics (generated)
├── data/
│   └── RF_shuffled_data.xlsx # Dataset (excluded from repo)
├── frontend/
│   ├── index.html          # User interface
│   ├── script.js           # Frontend logic
│   └── styles.css          # Styling
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```
