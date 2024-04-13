import os
import psycopg2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from datetime import datetime

def load_model_from_db(model_name, connection):
    select_query = "SELECT serialized_model FROM keras_models WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        serialized_model_data = result[0]
        model = load_model(serialized_model_data)
        return model
    else:
        print("Model not found.")
        return None


def make_predictions(model, input_date_str, data, sequence_length=100):
    # Preprocess data for prediction
    data_to_predict = data['close'].values[-sequence_length:]
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
model_name = "bidirectional_lstm_model"
input_date_str = '2027-02-09'  # Example date in YYYY-MM-DD format

# Load the model from the database
loaded_model = load_model_from_db(model_name, connection)

# Fetch historical stock data up to the input date
query = "SELECT * FROM stock_data WHERE date <= %s ORDER BY date ASC"
df = pd.read_sql(query, connection, params=[input_date_str], parse_dates=['date'])

# Make predictions if model loaded successfully and data fetched
if loaded_model and not df.empty:
    predicted_close_value = make_predictions(loaded_model, input_date_str, df, sequence_length=100)
    if predicted_close_value is not None:
        print(f"Predicted Close Value for {input_date_str}: {predicted_close_value}")

# Close the database connection
connection.close()
