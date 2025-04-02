# Wind Power Forecasting API for Orkney

This project is a RESTful API for predicting wind power using a trained **Random Forest** model. Built with **FastAPI**, it is deployed on an **Azure Virtual Machine**, and provides an interactive **Swagger UI** for easy testing.

The **Orkney Islands**, located in Northern Scotland, have significant wind and marine energy resources. Local farms can utilize wind power for energy generation. This app aims to use weather forecasting data to predict energy production for Orkney.


## 🌐 Live Demo

You can access the live API via:

👉 [http://20.234.40.109:8000/docs#/](http://20.234.40.109:8000/docs#/)

<img src="draw/api.png" alt="Swagger UI Screenshot" width="600">


## 📌 Available Endpoints

- **POST /predict**  
  Predict wind power based on user-provided meteorological input.

- **GET /predict/next-24h**  
  Predict wind power output for the next 24 hours.

- **GET /sample-input**  
  Returns a sample input format for testing.

- **GET /health**  
  Health check endpoint to verify API status.

## ⚙️ Tech Stack

- **Backend**: FastAPI (Python)
- **ML Models**: XGBoost, Linear Regression, Random Forest *(best performance and selected for deployment)*
- **Deployment**: Azure Virtual Machine (Ubuntu)
- **Documentation**: Swagger (via FastAPI’s auto-generated docs)
- **ML Ops Tools**: MLflow, Pandas, Scikit-learn
- **CI/CD**:
  - Docker for containerization and environment consistency  
  - GitHub Actions for automatic deployment to Azure VM on each `git push`


## 🧪 How to Run the Code and Train Models Locally

This program supports two main functionalities: **model training** and **prediction**.

1. Create the conda environment:

   ```bash
   conda env create -f conda_env.yaml
   conda activate windpower_prediction_Orkney
   ```

2. To train a model:

   ```bash
   mlflow run . --experiment-name windModel
   ```

   or

   ```bash
   python trainModel.py
   ```

3. To make predictions using the trained model:

   ```bash
   python predictWindpower.py
   ```


## 🌍 Data Sources

- **Wind power generation**:  
  [Scottish and Southern Electricity Networks (SSEN)](https://www.ssen.co.uk/)

- **Weather forecasts**:  
  [UK Met Office](https://www.metoffice.gov.uk/)


## 🤖 Model Training

The project uses traditional machine learning models from Scikit-learn to train on historical wind power and weather data:

- Linear Regression  
- XGBoost  
- Random Forest  

### Features used for training:

- Windspeed  
- Wind direction  
- Month  
- Day of the week  
- Hour  


## 📊 Model Selection & Observation

After training multiple models with varying parameters (30–365 day intervals and 3–10 time series splits), **Random Forest** performed best with:

- **180-day data interval**  
- **5-fold time series split**

The final model automatically retrieves and trains on the past 180 days of Orkney's wind power + weather data, and predicts the next **30 days** of generation.

## 📈 Main Results

<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">

windspeed and windpower visualizaion

<img src="draw/wind_speed_and_power_generation_2023.png" alt="Description" width="400"> 


wind direction and windpower visualizaion

<img src="draw/wind_direction and power generation in 2023.png" alt="Description" width="400">    


metrics

<img src="draw/metrics.png" alt="Description" width="400">  


model trained

<img src="draw/model_predictions-180d&5f.png" alt="Description" width="400">  


prediction with the best model

<img src="draw/wind_power_prediction_30days.png" alt="Description" width="400">    
</div>



