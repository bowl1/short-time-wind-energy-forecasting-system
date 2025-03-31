import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://20.234.40.109:5000")  # Azure IP
mlflow.set_experiment("WindPower")

model = joblib.load("saved_model/RandomForest.pkl")

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForest"
    )