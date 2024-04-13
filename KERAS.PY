import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime

def make_predictions(model_path, input_date_str, sequence_length=100):
    # Convert input date string from DD/MM/YYYY to YYYY-MM-DD format
    input_date = datetime.strptime(input_date_str, "%d/%m/%Y").strftime("%Y-%m-%d")

    # Load environment variables
    load_dotenv()

    # Get database connection details from environment variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    # Create a PostgreSQL connection
    connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    # Load the Keras model from the native Keras format file
    model = load_model(model_path)

  # Execute a query to fetch data
    query = "SELECT * FROM stock_data WHERE date <= %s ORDER BY date ASC"
    df = pd.read_sql(query, connection, params=[input_date], parse_dates=['date'])  # Use input_date instead of input_date_str


    # Close the database connection
    connection.close()

    # Parse input date
    input_date = pd.to_datetime(input_date_str, dayfirst=True)

    # Ensure there is enough historical data for prediction
    if len(df) < sequence_length:
        print(f"Not enough historical data for prediction.")
        return

    # Extract the close values for prediction
    data_to_predict = df['close'].values[-sequence_length:]

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
model_path = '/Users/moizibrar/work/PgAi/Save.keras'
input_date_str = '11/02/2020'

make_predictions(model_path, input_date_str)
