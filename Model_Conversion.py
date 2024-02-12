import os
import psycopg2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from datetime import datetime

def save_model_to_db(model, model_name, connection):
    # Serialize the model
    model_serialized = pickle.dumps(model)
    
    # Save the serialized model into PostgreSQL
    insert_query = """
    INSERT INTO model_storage (model_name, model_data) VALUES (%s, %s)
    ON CONFLICT (model_name) DO UPDATE SET model_data = excluded.model_data;
    """
    
    with connection.cursor() as cursor:
        cursor.execute(insert_query, (model_name, model_serialized))
        connection.commit()

def load_model_from_db(model_name, connection):
    select_query = """
    SELECT model_data FROM model_storage WHERE model_name = %s;
    """
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        model_data = pickle.loads(result[0])
        return model_data
    else:
        print("Model not found.")
        return None



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

# Path to your Keras model (adjust as necessary)
model_path = '/Users/moizibrar/work/finalfyp/Save.keras'

# Load and save your model to the database
keras_model = load_model(model_path)
save_model_to_db(keras_model, "Stock Prediction LSTM Model", connection)

# Load the model from the database for predictions
loaded_model = load_model_from_db("Stock Prediction LSTM Model", connection)




def check_model_inserted(model_name, connection):
    select_query = "SELECT COUNT(*) FROM model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result[0] > 0:
        print(f"Model '{model_name}' is present in the database.")
    else:
        print(f"Model '{model_name}' is not found in the database.")

# Example usage
check_model_inserted("Stock Prediction LSTM Model", connection)

def check_model_details(model_name, connection):
    select_query = "SELECT model_name, octet_length(model_data) AS data_size, inserted_at FROM model_storage WHERE model_name = %s;"
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        print(f"Model '{result[0]}' is present in the database. Data size: {result[1]} bytes. Inserted at: {result[2]}")
    else:
        print(f"Model '{model_name}' is not found in the database.")

# Example usage
check_model_details("Stock Prediction LSTM Model", connection)

# Close the database connection
connection.close()