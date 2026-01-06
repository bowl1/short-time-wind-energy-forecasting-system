import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from influxdb import InfluxDBClient

settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}
client = InfluxDBClient(
    host=settings["host"],
    port=settings["port"],
    username=settings["username"],
    password=settings["password"],
)
client.switch_database("orkney")


# Get the data and process it into DataFrame
def fetch_data(days=180):
    query = f"""
    SELECT * FROM "MetForecasts"
    WHERE time > now() - {days}d AND Lead_hours = '1'
    """
    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    df['time'] = pd.to_datetime(df['time'])
    return df


def encode_wind_direction(df):
    direction_to_radians = {
        "N": 0,
        "NNE": np.pi / 8,
        "NE": np.pi / 4,
        "ENE": 3 * np.pi / 8,
        "E": np.pi / 2,
        "ESE": 5 * np.pi / 8,
        "SE": 3 * np.pi / 4,
        "SSE": 7 * np.pi / 8,
        "S": np.pi,
        "SSW": 9 * np.pi / 8,
        "SW": 5 * np.pi / 4,
        "WSW": 11 * np.pi / 8,
        "W": 3 * np.pi / 2,
        "WNW": 13 * np.pi / 8,
        "NW": 7 * np.pi / 4,
        "NNW": 15 * np.pi / 8,
    }

    df["Direction_radians"] = df["Direction"].map(direction_to_radians)
    df["Direction_sin"] = np.sin(df["Direction_radians"])
    df["Direction_cos"] = np.cos(df["Direction_radians"])
    return df


# Get input data for the next 30 days
def prepare_data_for_prediction():
    df = fetch_data(180)  # Get 180 days of data
    df = encode_wind_direction(df)

    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["hour"] = df["time"].dt.hour

    # feature column
    features = ["Speed", "Direction_sin", "Direction_cos", "month", "day_of_week", "hour"]
    
    # Data preprocessing: standardized numerical features
    scaler = StandardScaler()
    df[["Speed", "Direction_sin", "Direction_cos"]] = scaler.fit_transform(df[["Speed", "Direction_sin", "Direction_cos"]])

    # Extract features for prediction
    X_future = df[features]
    
    # Return scaler and normalized features
    return X_future, scaler, df


# Loading a model in MLflow
def load_mlflow_model(model_uri='runs:/4649533270f44f099f3865458347d2ba/traditional_model'):
    # Use model_uri to load the model
    model = mlflow.pyfunc.load_model(model_uri)
    return model


# Make predictions and plot
def predict_and_plot():
    # Get prepared data and normalizer
    X_future, scaler, df = prepare_data_for_prediction()
    model = load_mlflow_model()

    # make predictions
    predictions_scaled = model.predict(X_future)

    # Denormalization: restore original scale
    predictions_original = predictions_scaled * scaler.scale_[0] + scaler.mean_[0]  # Restore the original value of wind speed
    predictions_original = predictions_original[:30]  # Cut the first 30 predicted values

    # Generate dates 30 days in the future
    future_dates = pd.date_range(start=pd.to_datetime("today"), periods=30, freq="D")

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, predictions_original, label="Predicted Wind Power", color='b')
    plt.xlabel("Date")
    plt.ylabel("Predicted Wind Power (MW)")
    plt.title("Wind Power Prediction for the Next 30 Days")
    plt.xticks(rotation=45)  
    plt.legend()
    plt.tight_layout() 
    plt.savefig("wind_power_prediction_30days.png")
    plt.show()


# Call functions for prediction and plotting
predict_and_plot()