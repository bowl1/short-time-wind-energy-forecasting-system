from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import math
import joblib
import os

app = FastAPI(
    title="Wind Power Forecast API",
    description="This API provides short-term wind power predictions based on input features such as wind speed, wind direction, and timestamp. It uses a pre-trained machine learning model to estimate wind energy output and supports both single-point and 24 hours predictions.",
    version="1.0.0",
)

# === Load Local Model ===
local_model_path = os.getenv("Local model path", "../train-model/saved_model/RandomForest.pkl")

try:
    model = joblib.load(local_model_path)
except Exception as e:
    print(f"Failed to load model from {local_model_path}: {e}")
    model = None


# === Request Schema ===
class InputData(BaseModel):
    # Do not cap with le=40 to avoid Pydantic 422; business rule enforced in handler
    wind_speed: float = Field(
        ..., ge=0, description="Wind speed (m/s), trained range 0–40; >40 returns 400"
    )
    wind_direction: float = Field(
        ..., ge=0, le=360, description="Wind direction (degrees), between 0–360"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description='ISO 8601 timestamp, e.g. "2025-03-30T12:00:00"; if omitted, current time is used',
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "wind_speed": 12.3,
                "wind_direction": 210,
                "timestamp": "2025-03-30T14:06:02",
            }
        }
    }


# === Predict One ===
@app.post(
    "/Predict/next 1h",
    responses={
        400: {"description": "Bad Request (e.g., wind speed out of trained range)"},
        503: {"description": "Service Unavailable (model not loaded)"},
        500: {"description": "Internal Server Error"},
    },
)
def predict_power(data: InputData):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded. Check LOCAL_MODEL_PATH: {local_model_path}")
        if data.wind_speed > 40:
            raise HTTPException(status_code=400, detail="Wind speed exceeds model trained range (0–40 m/s)")
        dt = data.timestamp or datetime.now()
        direction_rad = math.radians(data.wind_direction)
        warning = None
        if data.wind_speed > 25:
            warning = "Wind speed exceeds typical turbine cut-out speed"
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
        resp = {"Forecast kw": round(float(prediction[0]), 2), "Input": features}
        if warning:
            resp["warning"] = warning
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Predict 24 Hours ===
@app.post(
    "/Predict/next 24h",
    responses={
        400: {"description": "Bad Request (e.g., wind speed out of trained range)"},
        503: {"description": "Service Unavailable (model not loaded)"},
        500: {"description": "Internal Server Error"},
    },
)
def predict_next_24h(data: InputData):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded. Check LOCAL_MODEL_PATH: {local_model_path}")
        if data.wind_speed > 40:
            raise HTTPException(status_code=400, detail="Wind speed exceeds model trained range (0–40 m/s)")
        direction_rad = math.radians(data.wind_direction)
        direction_sin = math.sin(direction_rad)
        direction_cos = math.cos(direction_rad)
        now = datetime.now()
        warning = None
        if data.wind_speed > 25:
            warning = "Wind speed exceeds typical turbine cut-out speed"
        inputs = []
        for i in range(24):
            future_time = now + timedelta(hours=i)
            inputs.append(
                {
                    "Speed": data.wind_speed,
                    "Direction_sin": direction_sin,
                    "Direction_cos": direction_cos,
                    "month": future_time.month,
                    "day_of_week": future_time.weekday(),
                    "hour": future_time.hour,
                }
            )

        df = pd.DataFrame(inputs)
        predictions = model.predict(df)
        resp = {
            "Timestamps": [(now + timedelta(hours=i)).isoformat() for i in range(24)],
            "Forecast kw": [round(float(p), 2) for p in predictions],
        }
        if warning:
            resp["warning"] = warning
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Sample Input ===
@app.get("/Sample input")
def sample_input():
    return {
        "Wind speed": 12.3,
        "Wind direction": 210,
        "Timestamp": "2025-03-30T14:06:02",
    }


# === Health Check ===
@app.get("/Health")
def health_check():
    return {"Status": "Ok"}


# === Redirect root to Swagger UI ===
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/Docs")
