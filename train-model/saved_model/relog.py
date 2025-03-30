import joblib
import mlflow.sklearn
import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("file:/Users/libowen/github/windpower_prediction_Orkney/mlruns")

model = joblib.load("/Users/libowen/github/windpower_prediction_Orkney/saved_model/RandomForest.pkl") 

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForest"
    )