CREATE OR REPLACE FUNCTION get_db_config()
RETURNS TABLE(key TEXT, value TEXT) AS $$
DECLARE
    config RECORD;
BEGIN
    FOR config IN SELECT * FROM db_config LOOP
        RETURN NEXT config;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


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
import pickle

def get_db_config():
    # Use plpy to execute a query to get the database configuration
    result = plpy.execute("SELECT key, value FROM db_config;")
    config = {row['key']: row['value'] for row in result}
    return config

def load_model_and_scaler_from_db(model_name, connection):
    select_query = "SELECT model_data, scaler_data FROM apple_model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result is None or result[0] is None or result[1] is None:
        plpy.error(f"Model or scaler not found for model name: {model_name}.")
        return None, None
    
    model_data, scaler_data = result
    
    # Create a temporary file to write the model data
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(model_data)
    
    # Load the model from the temporary file
    try:
        model = load_model(temp_file_path)
    except Exception as e:
        os.remove(temp_file_path)
        plpy.error(f"Error loading model: {str(e)}")
    
    os.remove(temp_file_path)  # Delete the temporary file

    # Load the scaler
    try:
        scaler = pickle.loads(scaler_data)
    except Exception as e:
        plpy.error(f"Error loading scaler: {str(e)}")

    return model, scaler

def predict_future_prices(model, scaler, df, future_date, look_back=60):
    # Preprocess data
    close_data = df['close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaled_data = scaler.transform(close_data)

    # Start with the last look_back days from the scaled data
    last_look_back_data = scaled_data[-look_back:]

    predictions = []
    
    # Calculate the number of future days to predict
    last_date = pd.to_datetime(df.index[-1])
    future_date = pd.to_datetime(future_date)
    future_days = (future_date - last_date).days
    
    if future_days <= 0:
        plpy.error("Future date must be later than the last date in the dataset.")
        return None

    for _ in range(future_days):
        # Prepare the input data
        input_data = np.reshape(last_look_back_data, (1, look_back, 1))
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Append the prediction to the predictions list
        predictions.append(prediction[0, 0])
        
        # Update the input data with the new prediction
        last_look_back_data = np.append(last_look_back_data[1:], prediction, axis=0)

    # Invert the normalization of the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return float(predictions[-1])

def make_predictions(model, scaler, input_date_str, connection, sequence_length=60):
    # Format and validate input date
    input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
    
    # Fetch historical stock data up to the input date
    query = "SELECT date, close FROM apple_stock ORDER BY date ASC"
    df = pd.read_sql(query, connection, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    if len(df) < sequence_length:
        plpy.error("Not enough historical data for prediction.")
        return None
    
    # Predict future prices
    predicted_close_value = predict_future_prices(model, scaler, df, input_date_str, look_back=sequence_length)
    
    return predicted_close_value

# Fetch database connection details from the db_config table
db_config = get_db_config()

# Establish database connection using fetched details
connection = psycopg2.connect(dbname=db_config['db_name'], user=db_config['db_user'], password=db_config['db_password'], host=db_config['db_host'], port=int(db_config['db_port']))

# Specify the model name
model_name = "apple_lstm_model"

# Load the model and scaler from the database
loaded_model, loaded_scaler = load_model_and_scaler_from_db(model_name, connection)

# Make predictions if model and scaler loaded successfully
if loaded_model and loaded_scaler:
    predicted_close_value = make_predictions(loaded_model, loaded_scaler, input_date_str, connection, sequence_length=60)
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

CREATE OR REPLACE FUNCTION predict_stock_close_value_tesla(input_date_str TEXT)
RETURNS FLOAT AS $$
import os
import psycopg2
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def get_db_config():
    # Use plpy to execute a query to get the database configuration
    result = plpy.execute("SELECT key, value FROM db_config;")
    config = {row['key']: row['value'] for row in result}
    return config

def load_model_and_scaler_from_db(model_name, connection):
    select_query = "SELECT model_data, scaler_data FROM tesla_model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result is None or result[0] is None or result[1] is None:
        plpy.error(f"Model or scaler not found for model name: {model_name}.")
        return None, None
    
    model_data, scaler_data = result
    model = pickle.loads(model_data)
    scaler = pickle.loads(scaler_data)
    return model, scaler

def fetch_historical_data(connection):
    query = "SELECT * FROM tesla_stock ORDER BY date ASC"
    df = pd.read_sql(query, connection, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def make_predictions(model, scaler, dates, connection):
    # Create a DataFrame for the input dates
    date_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    date_df.set_index('date', inplace=True)
    
    # Fetch historical stock data
    historical_data = fetch_historical_data(connection)
    
    if historical_data.empty:
        plpy.error("No historical data available for prediction.")
        return None
    
    # Feature engineering on historical data
    historical_data['Year'] = historical_data.index.year
    historical_data['Month'] = historical_data.index.month
    historical_data['Day'] = historical_data.index.day
    historical_data['DayOfWeek'] = historical_data.index.dayofweek
    historical_data['MA10'] = historical_data['close'].rolling(window=10).mean()
    historical_data['MA50'] = historical_data['close'].rolling(window=50).mean()
    historical_data['MA200'] = historical_data['close'].rolling(window=200).mean()
    historical_data['Volatility'] = historical_data['close'].rolling(window=10).std()
    
    # Ensure no NaN values by filling with the last available values
    historical_data.fillna(method='ffill', inplace=True)
    historical_data.dropna(inplace=True)
    
    # Predicting for future dates
    future_data = pd.DataFrame(index=date_df.index)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data['Day'] = future_data.index.day
    future_data['DayOfWeek'] = future_data.index.dayofweek
    
    # Assume the future rolling averages and volatility are the same as the last available historical data
    last_row = historical_data.iloc[-1]
    future_data['MA10'] = last_row['MA10']
    future_data['MA50'] = last_row['MA50']
    future_data['MA200'] = last_row['MA200']
    future_data['Volatility'] = last_row['Volatility']
    
    # Define the features
    X_dates = future_data[['Year', 'Month', 'Day', 'DayOfWeek', 'MA10', 'MA50', 'MA200', 'Volatility']]
    
    # Standardize the features
    X_dates_scaled = scaler.transform(X_dates)
    
    # Predict the close prices
    predictions = model.predict(X_dates_scaled)
    
    # Add predictions to DataFrame
    future_data['Predicted_Close'] = predictions
    
    return future_data

# Fetch database connection details from the db_config table
db_config = get_db_config()

# Establish database connection using fetched details
connection = psycopg2.connect(dbname=db_config['db_name'], user=db_config['db_user'], password=db_config['db_password'], host=db_config['db_host'], port=int(db_config['db_port']))

# Specify the model name
model_name = "tesla_gradient_boosting_model"

# Load the model and scaler from the database
loaded_model, loaded_scaler = load_model_and_scaler_from_db(model_name, connection)

# Make predictions if model and scaler loaded successfully
if loaded_model and loaded_scaler:
    # Convert input_date_str to list of dates
    dates = [input_date_str]
    future_data = make_predictions(loaded_model, loaded_scaler, dates, connection)
    if future_data is not None and not future_data.empty:
        predicted_close_value = future_data.iloc[0]['Predicted_Close']

# Close the database connection
connection.close()

if predicted_close_value is not None:
    return predicted_close_value
else:
    plpy.error("Prediction could not be made.")
$$ LANGUAGE plpython3u;

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



-- Function to predict using ARIMA model for a specific date
-- Function to predict using ARIMA model for a specific date
CREATE OR REPLACE FUNCTION get_arima_prediction(target_date TEXT)
RETURNS FLOAT
AS $$
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import psycopg2

def get_db_config():
    # Use plpy to execute a query to get the database configuration
    result = plpy.execute("SELECT key, value FROM db_config;")
    config = {row['key']: row['value'] for row in result}
    return config

# Function to load the model from PostgreSQL
def load_model_from_db(model_name, connection):
    query = "SELECT model_data FROM google_model_storage WHERE model_name = %s;"
    cursor = connection.cursor()
    cursor.execute(query, (model_name,))
    result = cursor.fetchone()
    cursor.close()
    
    if result is None:
        raise ValueError(f"Model with name '{model_name}' not found in database.")
    
    model_data = result[0]
    model = pickle.loads(model_data)
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

# Fetch database connection details from the db_config table
db_config = get_db_config()

# Establish database connection using fetched details
connection = psycopg2.connect(dbname=db_config['db_name'], user=db_config['db_user'], password=db_config['db_password'], host=db_config['db_host'], port=int(db_config['db_port']))

# Fetch historical data from the PostgreSQL table
query = "SELECT date, close FROM google_stock ORDER BY date"
df = pd.read_sql(query, connection, parse_dates=['date'])
df.set_index('date', inplace=True)

# Load the ARIMA model from the database
model_name = 'sarimax_google_stock_model'
loaded_model = load_model_from_db(model_name, connection)

# Get the prediction for the target date
prediction = get_prediction_for_date(loaded_model, df, target_date)

# Close the database connection
connection.close()

return float(prediction)
$$ LANGUAGE plpython3u;

-- Function to get Google's stock data along with the predicted close value
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
    predicted_close_value FLOAT;
BEGIN
    -- Call get_arima_prediction function to get the predicted close value
    SELECT get_arima_prediction(input_date_str) INTO predicted_close_value;

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
            g.date,
            g.open::DOUBLE PRECISION,
            g.high::DOUBLE PRECISION,
            g.low::DOUBLE PRECISION,
            g.close::DOUBLE PRECISION,
            g.volume::DOUBLE PRECISION,
            g.adj_close::DOUBLE PRECISION,
            predicted_close_value as close_pred
        FROM
            google_stock g
        WHERE
            g.date = input_date_str::DATE;
    END IF;
END;
$$ LANGUAGE plpgsql;
