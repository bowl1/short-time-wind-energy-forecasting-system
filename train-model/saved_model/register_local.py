import joblib
import mlflow.sklearn
import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("file:/Users/libowen/github/windpower_prediction_Orkney/mlruns")

model = joblib.load("/Users/libowen/github/windpower_prediction_Orkney/train-model/saved_model/mlruns/925431245205846442/a7e549dcaa24498399e3acb332cc42ca/artifacts/traditional_model/model.pkl") 

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForest"
    )