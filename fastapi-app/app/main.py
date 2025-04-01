from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import math
import joblib
import os

app = FastAPI(
    title="Wind Power Forecast API",
    description="API for predicting wind power using a locally saved RandomForest model.",
    version="1.0.0",
)

# === Load Local Model ===
local_model_path = os.getenv("LOCAL_MODEL_PATH", "train-model/saved_model/RandomForest.pkl")

try:
    model = joblib.load(local_model_path)
except Exception as e:
    print(f"Failed to load model from {local_model_path}: {e}")
    model = None


# === Request Schema ===
class InputData(BaseModel):
    wind_speed: float
    wind_direction: float
    timestamp: str  # e.g., "2025-03-30T12:00:00"


# === Predict One ===
@app.post("/predict")
def predict_power(data: InputData):
    try:
        dt = datetime.fromisoformat(data.timestamp)
        direction_rad = math.radians(data.wind_direction)
        features = {
            "Speed": data.wind_speed,
            "Direction_sin": math.sin(direction_rad),
            "Direction_cos": math.cos(direction_rad),
            "month": dt.month,
            "day_of_week": dt.weekday(),
            "hour": dt.hour,
        }
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return {"forecast_kw": round(float(prediction[0]), 2), "input": features}
    except Exception as e:
        return {"error": str(e)}


# === Predict 24 Hours ===
@app.get("/predict/next-24h")
def predict_next_24h():
    try:
        now = datetime.now()
        inputs = []
        for i in range(24):
            future_time = now + timedelta(hours=i)
            inputs.append(
                {
                    "Speed": 10.0,  # 默认风速
                    "Direction_sin": 0.5,
                    "Direction_cos": 0.87,
                    "month": future_time.month,
                    "day_of_week": future_time.weekday(),
                    "hour": future_time.hour,
                }
            )

        df = pd.DataFrame(inputs)
        predictions = model.predict(df)
        return {
            "timestamps": [(now + timedelta(hours=i)).isoformat() for i in range(24)],
            "forecast_kw": [round(float(p), 2) for p in predictions],
        }
    except Exception as e:
        return {"error": str(e)}


# === Sample Input ===
@app.get("/sample-input")
def sample_input():
    return {
        "wind_speed": 12.3,
        "wind_direction": 210,
        "timestamp": "2025-03-30T14:06:02",
    }


# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok"}


# === Redirect root to Swagger UI ===
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")