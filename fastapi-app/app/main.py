from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import math
import os
import mlflow
import mlflow.pyfunc
from mlflow.models import Model
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
import time

# === Load .env ===
# load_dotenv()

app = FastAPI(
    title="Wind Power Forecast API",
    description="A REST API for predicting wind power using an MLflow-trained model. Includes prediction besed on input, next 24 hours prediction, metrics, and model info.",
    version="1.0.0",
)

# === Model Setup ===
use_local = os.getenv("USE_LOCAL_MLFLOW", "true").lower() == "true"
model = None
model_uri = ""
model_name = ""
client = None

if use_local:
    local_path = os.getenv("LOCAL_MODEL_PATH")
    model_uri = f"file:/app/{local_path}"
    model = mlflow.pyfunc.load_model(model_uri)
else:
    time.sleep(10)  # 等待 MLflow Server 启动完成（根据需要调整秒数）
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_name = os.getenv("MODEL_NAME")
    model_stage = os.getenv("MODEL_STAGE")
    model_uri = f"models:/RandomForest/Production"
    client = MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri)


# === Request Schema ===
class InputData(BaseModel):
    wind_speed: float
    wind_direction: float
    timestamp: str  # e.g., "2025-03-30T12:00:00"


# === Predict ===
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


# === Predict 24h Endpoint ===
@app.get("/predict/next-24h")
def predict_next_24h():
    try:
        now = datetime.now()
        inputs = []
        for i in range(24):
            future_time = now + timedelta(hours=i)
            inputs.append(
                {
                    "Speed": 10.0,
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


# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok"}


# === Model Info ===
@app.get("/info")
def model_info():
    try:
        if use_local:
            mlmodel = Model.load(os.path.join(local_path, "MLmodel"))
            return {
                "source": "local",
                "path": local_path,
                "flavors": list(mlmodel.flavors.keys()),
            }
        else:
            latest_version = client.get_latest_versions(
                model_name, stages=[model_stage]
            )[0]
            registry_path = mlflow.pyfunc.get_model_path(model_uri)
            mlmodel = Model.load(os.path.join(registry_path, "MLmodel"))
            return {
                "source": "mlflow registry",
                "model_name": model_name,
                "stage": model_stage,
                "version": latest_version.version,
                "run_id": latest_version.run_id,
                "flavors": list(mlmodel.flavors.keys()),
                "path": registry_path,
            }
    except Exception as e:
        return {"error": str(e)}


# === Model Metrics ===
@app.get("/metrics")
def model_metrics():
    if use_local:
        return {"message": "Metrics not available for local model."}
    else:
        latest_version = client.get_latest_versions(model_name, stages=[model_stage])[0]
        run_id = latest_version.run_id
        metrics = client.get_run(run_id).data.metrics
        return metrics


# === Sample Input ===
@app.get("/sample-input")
def sample_input():
    return {
        "wind_speed": 12.3,
        "wind_direction": 210,
        "timestamp": "2025-03-30T14:06:02",
    }


# === Swagger redirect ===
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
