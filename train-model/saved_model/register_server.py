import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "a7e549dcaa24498399e3acb332cc42ca"
model_path = "traditional_model"
source_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{model_path}"

try:
    client.create_registered_model("RandomForest_azure")
except:
    pass  # 如果已存在，会报错，可以忽略

client.create_model_version(
    name="RandomForest_azure",
    source=source_path,
    run_id=run_id,
    description="Re-register from existing run"
)