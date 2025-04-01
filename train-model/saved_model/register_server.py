import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_name = "RandomForest"

# 如果模型还没注册过
try:
    client.create_registered_model(model_name)
except:
    pass

client.create_model_version(
    name=model_name,
    source=f"../../mlruns/{experiment_id}/{run_id}/artifacts/traditional_model",
    run_id=run_id,
    description="Re-registered from existing run with metrics"
)