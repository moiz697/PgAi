import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import h5py
from statsmodels.tsa.stattools import adfuller
from datetime import date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

# Create the database connection
engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Load data from the PostgreSQL table
query = "SELECT date, close FROM google_stock"
df = pd.read_sql(query, engine)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'close' is numeric
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])
df = df[np.isfinite(df['close'])]

# Check stationarity function
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistics: %f' % result[0])
    print('P-Value: %f' % result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] <= 0.05:
        print("Reject the Null Hypothesis: Data is stationary")
    else:
        print("Fail to reject the Null Hypothesis: Data is not stationary")

check_stationarity(df['close'])

# Model parameters
p = 2
d = 1
q = 2
P = 2
D = 1
Q = 2
s = 12

# Fit the SARIMAX model
model = sm.tsa.statespace.SARIMAX(df['close'], order=(p, d, q), seasonal_order=(P, D, Q, s))
fit_model = model.fit()
print(fit_model.summary())

# Forecasting
forecast_periods = 20
forecast = fit_model.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Get the date range for plotting
start_date = df.index[-1] + pd.DateOffset(days=1)  # Start one day after the last date in the dataset
forecast_dates = pd.date_range(start=start_date, periods=forecast_periods, freq='D')

# Ensure the lengths match
assert len(forecast_dates) == len(forecast_mean), "Forecast dates and forecast mean lengths do not match."

# Plotting historical data and forecasts
plt.figure(figsize=(14,7))
plt.plot(df.index, df['close'], label='Actual')
plt.plot(forecast_dates, forecast_mean.values, color='red', label='Forecast')
plt.fill_between(forecast_dates, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='Confidence Intervals')

plt.title('Google Stock Closing Prices - Forecast for Next 20 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Save the model using pickle and h5py
with h5py.File('arima_model.h5', 'w') as f:
    pickled_model = pickle.dumps(fit_model)
    f.create_dataset('model', data=np.void(pickled_model))

# Function to load the model from an HDF5 file
def load_model_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        model_data = f['model'][()]
        model = pickle.loads(model_data.tobytes())
    return model

# Function to get forecast for a specific date
def get_forecast_for_date(file_path, start_date, target_date):
    # Load the model
    loaded_model = load_model_from_h5(file_path)
    
    # Calculate the number of days between the start date and the target date
    days_diff = (target_date - start_date).days
    
    if days_diff <= 0:
        raise ValueError("Target date must be after the start date.")
    
    # Forecasting with the loaded model
    forecast = loaded_model.get_forecast(steps=days_diff)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    # Get the forecast value for the target date
    forecast_value = forecast_mean.iloc[-1]
    conf_int = forecast_conf_int.iloc[-1]
    
    return forecast_value, conf_int

# Example usage
# Set the start date (last date in the training data)
start_date = df.index[-1]

# Set the target date for which you want the forecast
target_date = pd.Timestamp('2024-05-28')

# Get the forecast for the target date
forecast_value, conf_int = get_forecast_for_date('arima_model.h5', start_date, target_date)
print(f"Forecast value for {target_date}: {forecast_value}")
print(f"95% Confidence interval: {conf_int}")

# Plotting historical data and forecasts from the loaded model
forecast_dates = pd.date_range(start=start_date + pd.DateOffset(days=1), periods=(target_date - start_date).days, freq='D')
forecast = fit_model.get_forecast(steps=(target_date - start_date).days)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

plt.figure(figsize=(14,7))
plt.plot(df.index, df['close'], label='Actual')
plt.plot(forecast_dates, forecast_mean.values, color='red', label='Forecast')
plt.fill_between(forecast_dates, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='Confidence Intervals')

plt.title('Google Stock Closing Prices - Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
