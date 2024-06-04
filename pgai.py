import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def make_predictions(model_path, csv_file_path, input_date_str, sequence_length=100):
    # Load the Keras model from the native Keras format file
    model = load_model(model_path)

    # Read your CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, parse_dates=['date'], dayfirst=True)  # Parse dates with day first

    # Parse input date
    input_date = pd.to_datetime(input_date_str, dayfirst=True)

    # Ensure the sequence_length is defined
    # sequence_length = 100  # Assuming your sequence length is 10 (adjust this based on your model)

    # Extract the historical data up to the input date
    historical_data = df[df['date'] <= input_date]

    # Check if there is enough historical data for prediction
    if len(historical_data) < sequence_length:
        print(f"Not enough historical data for prediction.")
    else:
        # Extract the close values for prediction
        data_to_predict = historical_data['close'].values[-sequence_length:]

        # Reshape and preprocess the data for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_predict_scaled = scaler.fit_transform(data_to_predict.reshape(-1, 1))

        # Ensure the input shape matches the model's expectations
        data_to_predict_scaled = np.reshape(data_to_predict_scaled, (1, sequence_length, 1))

        # Make predictions using the loaded model
        predictions = model.predict(data_to_predict_scaled)

        # Inverse transform the predictions to get the original scale
        predicted_close_value = scaler.inverse_transform(predictions.reshape(-1, 1))

        print(f"Predicted Close Value for {input_date_str}: {predicted_close_value[0, 0]}")

# Example usage
model_path = '/Users/umerkhurshid/work/final/Save.keras'
csv_file_path = '/Users/umerkhurshid/Downloads/apple_stockmay.csv'
input_date_str = '05/13/2024'

make_predictions(model_path, csv_file_path, input_date_str)
