import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://20.234.40.109:5001") 
mlflow.set_experiment("WindPower")

model = joblib.load("/home/bowen/short-time-wind-energy-forecasting-system/train-model/saved_model/mlruns/925431245205846442/a7e549dcaa24498399e3acb332cc42ca/artifacts/traditional_model/model.pkl", compress=3)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForest"
    )