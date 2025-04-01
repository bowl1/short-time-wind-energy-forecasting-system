import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 

client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_name = "RandomForest"
model_artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts/traditional_model"

# 只注册一次，如果已经存在可以跳过
try:
    client.create_registered_model(model_name)
except:
    pass

# 注册为新版本
client.create_model_version(
    name=model_name,
    source=model_artifact_path, 
    run_id=run_id,
    description="Re-registered on Azure VM to fix local path issue"
)