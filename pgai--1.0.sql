/* contrib/pg_stat_monitor/pg_stat_monitor--2.0.sql */

-- Ensure the script is sourced via CREATE EXTENSION
CREATE OR REPLACE FUNCTION pg_test(input_date_str TEXT)
RETURNS TABLE(date DATE, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, name TEXT, close_pred FLOAT) AS $$
DECLARE
    predicted_close_value FLOAT;
BEGIN
    -- Call predict_stock_close_value function to get the predicted close value
    predicted_close_value := predict_stock_close_value(input_date_str);

    -- Fetch stock data for the given date
    IF input_date_str::DATE > CURRENT_DATE THEN
        -- Return only the predicted close value for future dates
        RETURN QUERY
        SELECT
            NULL::DATE as date,
            NULL::DOUBLE PRECISION as open,
            NULL::DOUBLE PRECISION as high,
            NULL::DOUBLE PRECISION as low,
            NULL::DOUBLE PRECISION as close,
            NULL::DOUBLE PRECISION as volume,
            NULL::TEXT as name,
            predicted_close_value as close_pred;
    ELSE
        -- Return stock data along with the predicted close value for historical dates
        RETURN QUERY
        SELECT
            stock_data.date,
            stock_data.open::DOUBLE PRECISION,
            stock_data.high::DOUBLE PRECISION,
            stock_data.low::DOUBLE PRECISION,
            stock_data.close::DOUBLE PRECISION,
            stock_data.volume::DOUBLE PRECISION,
            stock_data.name,
            predicted_close_value as close_pred
        FROM
            stock_data
        WHERE
            stock_data.date = input_date_str::DATE;
    END IF;
END;
$$ LANGUAGE plpgsql;




CREATE OR REPLACE FUNCTION predict_stock_close_value(input_date_str TEXT)
RETURNS FLOAT AS $$
import os
import psycopg2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime

# Database connection details
db_host = "localhost"
db_port = "5432"
db_name = "postgres"
db_user = "umerkhurshid"
db_password = "postgres"

# Establish database connection
connection = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)

def load_model_from_db(model_name, connection):
    select_query = "SELECT model_data FROM model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        model_data = pickle.loads(result[0])
        return model_data
    else:
        print("Model not found.")
        return None

def make_predictions(model, input_date_str, connection, sequence_length=100):
    # Format and validate input date
    input_date = datetime.strptime(input_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Fetch historical stock data up to the input date
    query = "SELECT * FROM stock_data WHERE date <= %s ORDER BY date ASC"
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

# Specify the model name
model_name = "Stock Prediction LSTM Model"

# Load the model from the database
loaded_model = load_model_from_db(model_name, connection)

# Make predictions if model loaded successfully
if loaded_model:
    predicted_close_value = make_predictions(loaded_model, input_date_str, connection, sequence_length=100)
    if predicted_close_value is not None:
        return predicted_close_value

# Close the database connection
connection.close()
$$ LANGUAGE plpython3u;
