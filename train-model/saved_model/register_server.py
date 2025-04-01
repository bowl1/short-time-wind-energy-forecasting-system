import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")  # ← docker 容器内必须用服务名通信
client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_name = "RandomForest"
artifact_path = "traditional_model"

source = f"mlruns/{experiment_id}/{run_id}/artifacts/{artifact_path}"

# 注册一次
try:
    client.create_registered_model(model_name)
except:
    pass

client.create_model_version(
    name=model_name,
    source=source,
    run_id=run_id,
    description="Registering model from existing run"
)