import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ... (previous code)

def make_stock_predictions(model_path, csv_file_path, input_date_str, sequence_length=100):
    # Load the Keras model from the native Keras format file
    model = load_model(model_path)

    # Read your CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, parse_dates=['date'], dayfirst=True)

    # Parse input date
    input_date = pd.to_datetime(input_date_str, dayfirst=True)

    # Find the index corresponding to the input date in CSV file
    input_date_index_csv = np.where(df['date'] <= input_date)[0][-1]

    # Extract the historical data up to the input date from CSV file
    start_index_csv = max(0, input_date_index_csv - sequence_length + 1)
    data_to_predict_csv = df['close'].values[start_index_csv:input_date_index_csv + 1]

    # Reshape and preprocess the data for prediction in CSV file
    scaler_csv = MinMaxScaler(feature_range=(0, 1))
    data_to_predict_scaled_csv = scaler_csv.fit_transform(data_to_predict_csv.reshape(-1, 1))

    # Ensure the input shape matches the model's expectations for CSV file
    data_to_predict_scaled_csv = np.reshape(data_to_predict_scaled_csv, (1, sequence_length, 1))

    # Make predictions using the loaded model for CSV file
    predictions_csv = model.predict(data_to_predict_scaled_csv)

    # Inverse transform the predictions to get the original scale for CSV file
    predicted_close_value_csv = scaler_csv.inverse_transform(predictions_csv.reshape(-1, 1))

    print(f"Predicted Close Value from CSV for {input_date_str}: {predicted_close_value_csv[0, 0]:.2f}")

# ... (rest of the code)


# Example usage
model_path = '/Users/moizibrar/Work/fyp/Save.keras'  # Replace with the path to your .keras file
csv_file_path = '/Users/moizibrar/Downloads/archive/individual_stocks_5yr/individual_stocks_5yr/AAL_data.csv'  # Replace with the path to your CSV file
input_date_str = '01/05/2025'  # Replace with the desired input date (MM/DD/YYYY)

make_stock_predictions(model_path, csv_file_path, input_date_str)
