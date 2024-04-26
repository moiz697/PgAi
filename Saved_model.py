import os
import psycopg2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from keras.models import model_from_json
import json
from tensorflow.keras.models import model_from_json
from datetime import datetime
def load_model_from_db(model_name, connection):
    select_query = "SELECT model_data FROM msci_pak_global_model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        model_data = result[0]
        try:
            model_json_str = model_data.tobytes().decode('utf-8')  # Convert memoryview to bytes and decode to string
            model = model_from_json(model_json_str)
            return model
        except Exception as e:
            print("Error loading model:", e)
            return None
    else:
        print("Model not found.")
        return None


def make_predictions(model, input_date_str, connection, sequence_length=100):
    # Format and validate input date
    input_date = datetime.strptime(input_date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    
    # Fetch historical stock data up to the input date
    query = "SELECT * FROM msci_pak_global_stock WHERE date <= %s ORDER BY date ASC"
    df = pd.read_sql(query, connection, params=[input_date], parse_dates=['date'])
    
    if len(df) < sequence_length:
        print("Not enough historical data for prediction.")
        return None
    
    # Preprocess data for prediction
    data_to_predict = df['close'].values[-sequence_length:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_to_predict_scaled = scaler.fit_transform(data_to_predict.reshape(-1, 1))
    data_to_predict_scaled = np.reshape(data_to_predict_scaled, (1, sequence_length, 1))
    
    # Predict using the loaded model
    predictions = model.predict(data_to_predict_scaled)
    predicted_close_value = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    return predicted_close_value[0, 0]

# Load environment variables
load_dotenv()

# Database connection details
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# Establish database connection
connection = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)

# Specify the model name and input date for prediction
model_name = "Stock Prediction LSTM Model"
input_date_str = '24/03/2020'  # Example date

# Load the model from the database
loaded_model = load_model_from_db(model_name, connection)

# Make predictions if model loaded successfully
if loaded_model:
    predicted_close_value = make_predictions(loaded_model, input_date_str, connection, sequence_length=100)
    if predicted_close_value is not None:
        print(f"Predicted Close Value for {input_date_str}: {predicted_close_value}")

# Close the database connection
connection.close()
