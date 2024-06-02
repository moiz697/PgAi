import h5py
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_model_from_h5(file_path):
    print("Loading model from:", file_path)
    with h5py.File(file_path, 'r') as f:
        model_data = f['model'][()]
        model = pickle.loads(model_data.tobytes())
    print("Model loaded successfully.")
    return model

def get_prediction_for_date(model, historical_data, target_date):
    # Get the last date from the historical data
    start_date = historical_data.index[-1]
    
    # Calculate the number of days between the start date and the target date
    days_diff = (target_date - start_date).days
    
    if days_diff <= 0:
        raise ValueError("Target date must be after the start date.")
    
    # Forecasting with the loaded model
    forecast = model.get_forecast(steps=days_diff)
    forecast_mean = forecast.predicted_mean
    
    # Get the prediction value for the target date
    prediction_value = forecast_mean.iloc[-1]
    
    return prediction_value

# Example usage
# Load historical data
file_path = '/Users/moizibrar/Downloads/GOOG.csv'
df = pd.read_csv(file_path)
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])
df = df[np.isfinite(df['Close'])]

# Set the target date for which you want the prediction
target_date = pd.Timestamp('2024-06-05')

# Enter the full path to your .h5 file
file_path = '/Users/moizibrar/work/pgai/arima_model.h5'

# Load the model
loaded_model = load_model_from_h5(file_path)

# Get the prediction for the target date
prediction = get_prediction_for_date(loaded_model, df, target_date)
print(f"Prediction for {target_date}: {prediction}")
