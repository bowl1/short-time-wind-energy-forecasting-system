import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
client = MlflowClient()

run_id = "63cb92ead13b427ab68bde2d5cc13899"
model_path = "traditional_model"  # artifact 路径
experiment_id = "925431245205846442"

# 注册模型（如果没有注册过可以加上）
try:
    client.create_registered_model("RandomForest_azure")
except:
    pass  # 已经存在会报错，忽略即可

# 创建模型版本，引用原始 run 的 artifact
client.create_model_version(
    name="RandomForest",
    source=f"/home/bowen/short-time-wind-energy-forecasting-system/train-model/saved_model/mlruns/925431245205846442/a7e549dcaa24498399e3acb332cc42ca/artifacts/traditional_model",
    run_id=run_id,
    description="Re-registered from original training run  with metrics"
)