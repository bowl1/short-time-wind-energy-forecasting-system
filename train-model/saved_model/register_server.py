import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("WindPower")

model = joblib.load("/mlflow/mlruns/925431245205846442/63cb92ead13b427ab68bde2d5cc13899/artifacts/traditional_model/model.pkl")

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForest"
    )
    print("✅ Registered with new run_id:", run.info.run_id)