from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, AwareDatetime, field_validator, model_validator
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import math
import joblib
import os
from pathlib import Path
from prometheus_client import Gauge, REGISTRY, GC_COLLECTOR, PLATFORM_COLLECTOR, disable_created_metrics
from prometheus_client.core import GaugeMetricFamily
from threading import Lock
from prometheus_fastapi_instrumentator import Instrumentator, metrics

app = FastAPI(
    title="Wind Power Forecast API",
    description="This API provides short-term wind power predictions based on input features such as wind speed, wind direction, and timestamp. It uses a pre-trained machine learning model to estimate wind energy output and supports both single-point and 24 hours predictions.",
    version="1.0.0",
)

# === Prometheus Metrics ===
# Drop series nothing reads: *_created timestamps, Python GC and interpreter info.
# process_* (memory/CPU) stays for sizing the container's resource limits.
disable_created_metrics()
REGISTRY.unregister(GC_COLLECTOR)
REGISTRY.unregister(PLATFORM_COLLECTOR)

# Only the request counter and per-endpoint latency the dashboard uses.
# Patterns are regex searches, so the root redirect needs anchoring.
Instrumentator(excluded_handlers=["/metrics", "/Health", "/ready", "/docs", "/openapi.json", "/redoc", "^/$"]).add(
    metrics.requests(),
    metrics.latency(should_include_status=False, buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)),
).instrument(app).expose(app, include_in_schema=False)
MODEL_LOADED = Gauge("wind_model_loaded", "1 if the model loaded successfully, else 0")

# === Load Local Model ===
local_model_path = os.getenv(
    "LOCAL_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "train-model/saved_model/RandomForest.pkl"),
)

try:
    model = joblib.load(local_model_path)
except Exception as e:
    print(f"Failed to load model from {local_model_path}: {e}")
    model = None
MODEL_LOADED.set(1 if model is not None else 0)


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
        dt = data.timestamp or datetime.now(timezone.utc)
        # Legacy naive timestamps are interpreted as UTC.
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
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


# === Explicit hourly weather / 24-hour forecast ===
class WeatherHour(BaseModel):
    timestamp: AwareDatetime
    wind_speed: float = Field(ge=0, le=40, allow_inf_nan=False)
    wind_direction: float = Field(ge=0, le=360, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def utc_hour(cls, value):
        value = value.astimezone(timezone.utc)
        if value.minute or value.second or value.microsecond:
            raise ValueError("timestamp must be on a whole UTC hour")
        return value


class ForecastRequest(BaseModel):
    hours: list[WeatherHour] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def consecutive_hours(self):
        for previous, current in zip(self.hours, self.hours[1:]):
            if current.timestamp - previous.timestamp != timedelta(hours=1):
                raise ValueError("hours must be ordered, unique and exactly one hour apart")
        return self


def feature_row(hour):
    dt = hour.timestamp.astimezone(timezone.utc)
    angle = math.radians(hour.wind_direction)
    return {"Speed": hour.wind_speed, "Direction_sin": math.sin(angle),
            "Direction_cos": math.cos(angle), "month": dt.month,
            "day_of_week": dt.weekday(), "hour": dt.hour}


class LatestForecast:
    """One atomic snapshot; fixed 1..24 labels avoid timestamp label churn.

    Single process / single replica, matching this project's deployment.
    Samples are absent until a successful forecast, never fabricated zeros.
    """
    def __init__(self):
        self.lock = Lock()
        self.snapshot = None

    def publish(self, snapshot):
        with self.lock:
            if self.snapshot is None or snapshot["generated_at"] >= self.snapshot["generated_at"]:
                self.snapshot = snapshot

    def read(self):
        with self.lock:
            return self.snapshot

    def collect(self):
        snapshot = self.read()
        if snapshot is None:
            return
        generated = GaugeMetricFamily("wind_forecast_generated_timestamp_seconds",
                                      "Time the latest successful 24h batch was generated")
        generated.add_metric([], snapshot["generated_at"].timestamp())
        yield generated
        power = GaugeMetricFamily("wind_predicted_power_kw", "Latest batch power by hourly slot",
                                  labels=["horizon"])
        target = GaugeMetricFamily("wind_forecast_target_timestamp_seconds", "UTC target time by hourly slot",
                                   labels=["horizon"])
        for i, point in enumerate(snapshot["predictions"], 1):
            power.add_metric([str(i)], point["power_kw"])
            target.add_metric([str(i)], point["timestamp"].timestamp())
        yield power
        yield target


latest_forecast = LatestForecast()
REGISTRY.register(latest_forecast)


@app.post("/predict/1h")
def predict_hour(data: WeatherHour):
    # Same feature conventions and model as the batch endpoint.
    result = predict_power(InputData(**data.model_dump()))
    return {"timestamp": data.timestamp, "power_kw": result["Forecast kw"],
            **({"warning": result["warning"]} if "warning" in result else {})}


@app.post("/predict/24h")
def predict_next_24h(data: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    generated_at = datetime.now(timezone.utc)
    next_hour = generated_at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if data.hours[0].timestamp != next_hour:
        raise HTTPException(status_code=422, detail="First timestamp must be the next whole UTC hour")
    values = model.predict(pd.DataFrame([feature_row(hour) for hour in data.hours]))
    if len(values) != 24 or not all(math.isfinite(float(v)) for v in values):
        raise HTTPException(status_code=500, detail="Model returned invalid predictions")
    snapshot = {
        "generated_at": generated_at,
        "timezone": "UTC",
        "predictions": [{"timestamp": hour.timestamp, "power_kw": round(float(value), 2)}
                        for hour, value in zip(data.hours, values)],
    }
    if any(hour.wind_speed > 25 for hour in data.hours):
        snapshot["warning"] = "Some wind speeds exceed typical turbine cut-out speed"
    latest_forecast.publish(snapshot)
    return snapshot


@app.get("/forecast/latest")
def get_latest_forecast():
    snapshot = latest_forecast.read()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="No successful forecast yet")
    return {**snapshot, "age_seconds": (datetime.now(timezone.utc) - snapshot["generated_at"]).total_seconds()}


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


@app.get("/ready", include_in_schema=False)
def readiness_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"Status": "Ready"}


# === Redirect root to Swagger UI ===
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
