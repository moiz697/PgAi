import os
import psycopg2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, model_from_json
from dotenv import load_dotenv

# Function to save the model to the database
def save_model_to_db(model, model_name, connection):
    # Serialize the model to JSON
    model_json = model.to_json()
    
    # Encode the serialized model data as bytes
    model_data = model_json.encode('utf-8')
    
    # Save the encoded model data into PostgreSQL
    insert_query = """
    INSERT INTO msci_pak_global_model_storage    (model_name, model_data) VALUES (%s, %s)
    ON CONFLICT (model_name) DO UPDATE SET model_data = excluded.model_data;
    """
    
    with connection.cursor() as cursor:
        cursor.execute(insert_query, (model_name, model_data))
        connection.commit()

# Function to load the model from the database
def load_model_from_db(model_name, connection):
    select_query = """
    SELECT model_data FROM msci_pak_global_model_storage    WHERE model_name = %s;
    """
    
    with connection.cursor() as cursor:
        cursor.execute(select_query, (model_name,))
        result = cursor.fetchone()
    
    if result:
        model_json = result[0]
        # Ensure model_json is converted to string before deserialization
        if isinstance(model_json, bytes):
            model_json_str = model_json.decode('utf-8')
        elif isinstance(model_json, memoryview):
            model_json_str = model_json.tobytes().decode('utf-8')
        else:
            model_json_str = model_json  # Already a string
        
        model = model_from_json(model_json_str)
        return model
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
model_path = '/Users/moizibrar/Library/Mobile Documents/com~apple~CloudDocs/Downloads/umer code fyp/ msci_pak_global_stock.keras'

# Load your model
keras_model = load_model(model_path)

# Save your model to the database
save_model_to_db(keras_model, "Stock Prediction LSTM Model", connection)

# Load the model from the database
loaded_model = load_model_from_db("Stock Prediction LSTM Model", connection)

# Close the database connection
connection.close()
