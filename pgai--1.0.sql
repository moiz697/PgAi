/* contrib/pg_stat_monitor/pg_stat_monitor--2.0.sql */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgai" to load this file. \quit

CREATE FUNCTION apple_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'apple_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

CREATE FUNCTION tesla_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'tesla_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

CREATE FUNCTION msci_pak_global_stock (
out date    date,
out open    numeric,
out high    numeric,
out low     numeric,
out close   numeric,
out volume  bigint,

out close_pred    int
)
RETURNS SETOF record



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

AS 'MODULE_PATHNAME', 'msci_pak_global_stock'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;
