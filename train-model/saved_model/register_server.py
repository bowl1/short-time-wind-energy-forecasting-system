import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://20.234.40.109:5001") 

model_name = "RandomForest"
run_id = "63cb92ead13b427ab68bde2d5cc13899"
artifact_path = "traditional_model"

model_uri = f"runs:/{run_id}/{artifact_path}"

mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)