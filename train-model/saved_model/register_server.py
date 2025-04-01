import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_name = "RandomForest_azure"
model_artifact_path = "traditional_model"

# 如果还没注册过模型名
try:
    client.create_registered_model(model_name)
except:
    pass

# 重新注册为新版本（关键是下面这一行必须是绝对路径）
client.create_model_version(
    name=model_name,
    source=f"/home/bowen/short-time-wind-energy-forecasting-system/mlruns/{experiment_id}/{run_id}/artifacts/{model_artifact_path}",
    run_id=run_id,
    description="Re-registered from original run, with metrics & original artifact"
)