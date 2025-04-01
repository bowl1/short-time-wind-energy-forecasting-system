import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
client = MlflowClient()

experiment_id = "925431245205846442"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_name = "RandomForest_azure"
model_artifact_path = "traditional_model"

client = MlflowClient()

# 👇 注册模型（如果没有）
try:
    client.create_registered_model(model_name)
except:
    pass  # 已存在会报错，忽略即可

# ✅ 使用 run_id 和 artifact 路径重新注册版本
client.create_model_version(
    name=model_name,
    source=f"mlruns/{experiment_id}/{run_id}/artifacts/{model_artifact_path}",
    run_id=run_id,
    description="Re-registered from existing run with full metrics"
)