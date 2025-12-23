import requests
import os
from dotenv import load_dotenv
# 1. Load the secret key from .env (robustly handle script location)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

api_key = os.getenv("API_KEY")
# 2. Define the URL
url = "http://localhost:8000/predict"
# 3. Define the Headers (The "Key")
headers = {
    "x-api-key": api_key
}
# 4. Define the Data (The Car Features)
data = {
    "model_year": 2024,
    "engine_size": 2.5,
    "cylinders": 4,
    "fuel_type": "Z",         # Premium Gasoline
    "transmission": "AS6",    # Automatic with select shift
    "vehicle_class": "SUV"
}
try:
    # 5. Send the POST request
    response = requests.post(url, json=data, headers=headers)
    # 6. Check the result
    if response.status_code == 200:
        print("Success! Prediction:", response.json())
    elif response.status_code == 403:
        print("Failed: Invalid API Key!")
    else:
        print(f"Failed with Status {response.status_code}:", response.text)
        
except Exception as e:
    print("Could not connect to server. Is it running?")
    print(e)