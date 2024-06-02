-- Function to predict Apple's stock close value
-- Function to predict Apple's stock close value
CREATE OR REPLACE FUNCTION predict_stock_close_value_apple(input_date_str TEXT)
RETURNS FLOAT AS $$
import os
import psycopg2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
import tempfile

def load_model_from_db(model_name, connection):
    select_query = "SELECT model_data FROM apple_model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        model_data = result[0]
        # Create a temporary file to write the model data
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(model_data)
        # Load the model from the temporary file
        model = load_model(temp_file_path)
        os.remove(temp_file_path)  # Delete the temporary file
        return model
    else:
        plpy.error("Model not found.")
        return None

def make_predictions(model, input_date_str, connection, sequence_length=100):
    # Format and validate input date
    input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
    
    # Fetch historical stock data up to the input date
    query = "SELECT date, close FROM apple_stock WHERE date <= %s ORDER BY date ASC"
    df = pd.read_sql(query, connection, params=[input_date.strftime("%Y-%m-%d")], parse_dates=['date'])
    
    if len(df) < sequence_length:
        plpy.error("Not enough historical data for prediction.")
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

# Database connection details
db_host = 'localhost'
db_port = '5432'
db_name = 'postgres'
db_user = 'moizibrar'
db_password = 'postgres'

# Establish database connection
connection = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)

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
-- Function to get Apple's stock data along with the predicted close value
CREATE OR REPLACE FUNCTION apple_stock(input_date_str TEXT)
RETURNS TABLE(
    date DATE, 
    open DOUBLE PRECISION, 
    high DOUBLE PRECISION, 
    low DOUBLE PRECISION, 
    close DOUBLE PRECISION, 
    volume DOUBLE PRECISION, 
    adj_close DOUBLE PRECISION,  
    close_pred FLOAT
) AS $$
DECLARE
    predicted_close_value FLOAT;
BEGIN
    -- Call predict_stock_close_value_apple function to get the predicted close value
    SELECT predict_stock_close_value_apple(input_date_str) INTO predicted_close_value;

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
            NULL::DOUBLE PRECISION as adj_close,
            predicted_close_value as close_pred;
    ELSE
        -- Return stock data along with the predicted close value for historical dates
        RETURN QUERY
        SELECT
            a.date,
            a.open::DOUBLE PRECISION,
            a.high::DOUBLE PRECISION,
            a.low::DOUBLE PRECISION,
            a.close::DOUBLE PRECISION,
            a.volume::DOUBLE PRECISION,
            a.adj_close::DOUBLE PRECISION,
            predicted_close_value as close_pred
        FROM
            apple_stock a
        WHERE
            a.date = input_date_str::DATE;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION tesla_stock(input_date_str TEXT)
RETURNS TABLE(
    date DATE, 
    open DOUBLE PRECISION, 
    high DOUBLE PRECISION, 
    low DOUBLE PRECISION, 
    close DOUBLE PRECISION, 
    volume DOUBLE PRECISION, 
    adj_close DOUBLE PRECISION,  
    close_pred FLOAT
) AS $$
DECLARE
    predicted_close_value FLOAT;
BEGIN
    -- Call predict_stock_close_value_tesla function to get the predicted close value
    SELECT predict_stock_close_value_tesla(input_date_str) INTO predicted_close_value;

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
            NULL::DOUBLE PRECISION as adj_close,
            predicted_close_value as close_pred;
    ELSE
        -- Return stock data along with the predicted close value for historical dates
        RETURN QUERY
        SELECT
            t.date,
            t.open::DOUBLE PRECISION,
            t.high::DOUBLE PRECISION,
            t.low::DOUBLE PRECISION,
            t.close::DOUBLE PRECISION,
            t.volume::DOUBLE PRECISION,
            t.adj_close::DOUBLE PRECISION,
            predicted_close_value as close_pred
        FROM
            tesla_stock t
        WHERE
            t.date = input_date_str::DATE;
    END IF;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION predict_stock_close_value_tesla(input_date_str TEXT)
RETURNS FLOAT AS $$
import psycopg2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
from datetime import datetime

def load_model_from_db(model_name, connection):
    select_query = "SELECT model_data FROM tesla_model_storage WHERE model_name = %s;"
    
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
            plpy.error("Error loading model: {}".format(e))
            return None
    else:
        plpy.error("Model not found.")
        return None

def make_predictions(model, input_date_str, connection, sequence_length=100):
    # Format and validate input date
    input_date = datetime.strptime(input_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Fetch historical stock data up to the input date
    query = "SELECT * FROM tesla_stock WHERE date <= %s ORDER BY date ASC"
    df = pd.read_sql(query, connection, params=[input_date], parse_dates=['date'])
    
    if len(df) < sequence_length:
        plpy.error("Not enough historical data for prediction.")
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

# Database connection details
db_host = 'localhost'
db_port = '5432'
db_name = 'postgres'
db_user = 'moizibrar'
db_password = 'postgres'

# Establish database connection
connection = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)

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


CREATE OR REPLACE FUNCTION get_arima_prediction(target_date TEXT)
RETURNS FLOAT
AS $$
  import h5py
  import pandas as pd
  import numpy as np
  import pickle
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from datetime import datetime
  
  # Function to load ARIMA model from .h5 file
  def load_model_from_h5(file_path):
      with h5py.File(file_path, 'r') as f:
          model_data = f['model'][()]
          model = pickle.loads(model_data.tobytes())
      return model

  # Function to get the prediction for a specific date
  def get_prediction_for_date(model, historical_data, target_date):
      # Get the last date from the historical data
      start_date = historical_data.index[-1]
      
      # Convert target date string to Timestamp object
      target_date = pd.Timestamp(target_date)
      
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

  # Fetch historical data from the PostgreSQL table
  query = "SELECT date, close FROM google_stock ORDER BY date"
  result = plpy.execute(query)
  
  # Convert the result to a DataFrame
  data = {'date': [], 'close': []}
  for row in result:
      data['date'].append(row['date'])
      data['close'].append(row['close'])
  df = pd.DataFrame(data)
  df['date'] = pd.to_datetime(df['date'])
  df.set_index('date', inplace=True)
  
  # Load the ARIMA model from the specified path
  model_file_path = '/Users/moizibrar/work/pgai/arima_model.h5'  # Update this with the actual path
  loaded_model = load_model_from_h5(model_file_path)
  
  # Get the prediction for the target date
  prediction = get_prediction_for_date(loaded_model, df, target_date)
  
  return float(prediction)
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION google_stock(input_date_str TEXT)
RETURNS TABLE(
    date DATE, 
    open DOUBLE PRECISION, 
    high DOUBLE PRECISION, 
    low DOUBLE PRECISION, 
    close DOUBLE PRECISION, 
    volume DOUBLE PRECISION, 
    adj_close DOUBLE PRECISION,  
    close_pred FLOAT
) AS $$
DECLARE
    predicted_close FLOAT;
    historical_data RECORD;
BEGIN
    -- Fetch historical stock data for the given date
    SELECT
        g.date,
        g.open::DOUBLE PRECISION,
        g.high::DOUBLE PRECISION,
        g.low::DOUBLE PRECISION,
        g.close::DOUBLE PRECISION,
        g.volume::DOUBLE PRECISION,
        g.adj_close::DOUBLE PRECISION
    INTO
        historical_data
    FROM
        google_stock g
    WHERE
        g.date = input_date_str::DATE;

    -- Fetch the prediction value using get_arima_prediction function
    SELECT get_arima_prediction(input_date_str) INTO predicted_close;

    -- Return the fetched data along with the prediction value
    RETURN QUERY
    SELECT
        historical_data.date,
        historical_data.open,
        historical_data.high,
        historical_data.low,
        historical_data.close,
        historical_data.volume,
        historical_data.adj_close,
        predicted_close;
END;
$$ LANGUAGE plpgsql;
