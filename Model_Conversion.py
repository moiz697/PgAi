import os
import psycopg2
from tensorflow.keras.models import load_model, save_model
from dotenv import load_dotenv
import tempfile

# Function to save the model to the database
def save_model_to_db(model, model_name, connection):
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_file_path = temp_file.name
        save_model(model, temp_file_path)
        # Read the model file as binary data
        with open(temp_file_path, 'rb') as file:
            model_data = file.read()
    # Save the model data into PostgreSQL
    insert_query = """
    INSERT INTO tesla_model_storage (model_name, model_data) VALUES (%s, %s)
    ON CONFLICT (model_name) DO UPDATE SET model_data = excluded.model_data;
    """
    with connection.cursor() as cursor:
        cursor.execute(insert_query, (model_name, psycopg2.Binary(model_data)))
        connection.commit()

# Function to load the model from the database
def load_model_from_db(model_name, connection):
    select_query = """
    SELECT model_data FROM tesla_model_storage WHERE model_name = %s;
    """
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
model_path = '/Users/moizibrar/Downloads/final/Save.keras'

# Load your model
keras_model = load_model(model_path)

# Save your model to the database
save_model_to_db(keras_model, "Stock Prediction LSTM Model", connection)

# Load the model from the database
loaded_model = load_model_from_db("Stock Prediction LSTM Model", connection)

# Close the database connection
connection.close()
