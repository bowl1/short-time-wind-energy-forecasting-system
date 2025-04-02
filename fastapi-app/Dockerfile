FROM python:3.10

WORKDIR /app

COPY fastapi-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi-app/app/ ./app/
COPY train-model/saved_model/ ./train-model/saved_model/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]