
# 启动 MLflow Tracking Server
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1