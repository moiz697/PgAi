import os
import psycopg2
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import pickle

# Load environment variables from .env file
load_dotenv()

# Function to create the models table if not exists
def create_models_table(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_data BYTEA,
                stored_date DATE NOT NULL
            );
        """)
        print("Table created successfully")
    conn.commit()

# Function to insert model data into the database
def insert_model(conn, model_name, model_data, stored_date):
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO models (model_name, model_data, stored_date) VALUES (%s, %s, %s);
        """, (model_name, model_data, stored_date))
        print("Model inserted successfully")
    conn.commit()

# Function to deserialize model from database
def load_model_from_db(conn, model_name):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT model_data FROM models WHERE model_name = %s;
        """, (model_name,))
        model_bytes = cursor.fetchone()[0]

    # Deserialize the model
    model = pickle.loads(model_bytes)
    return model

# Function to check if data from a specific date is stored
def is_data_stored_for_date(conn, date_to_check):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) FROM models WHERE stored_date = %s;
        """, (date_to_check,))
        count = cursor.fetchone()[0]
    return count > 0

def main():
    # Get database connection details from environment variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    print("DB_HOST:", db_host)
    print("DB_PORT:", db_port)
    print("DB_NAME:", db_name)
    print("DB_USER:", db_user)
    print("DB_PASSWORD:", db_password)

    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )

    create_models_table(conn)

    model_file_path = '/Users/moizibrar/work/Save.h5'
    stored_date = '2024-04-08'  # Date to check

    # Insert the model into the database
    with open(model_file_path, 'rb') as f:
        model_data = f.read()
    insert_model(conn, os.path.basename(model_file_path), model_data, stored_date)

    # Check if data from a specific date is stored
    date_to_check = '2024-04-08'  # Date to check
    if is_data_stored_for_date(conn, date_to_check):
        print(f"Data is stored for the date: {date_to_check}")
    else:
        print(f"No data is stored for the date: {date_to_check}")

    conn.close()

if __name__ == "__main__":
    main()
