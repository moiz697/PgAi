import tensorflow as tf
from io import BytesIO
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label, Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Get database connection details from environment variables
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
model_table = os.getenv("MODEL_TABLE")
model_name = os.getenv("MODEL_NAME")

# Create a PostgreSQL connection
connection = psycopg2.connect(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_password
)

query = f"SELECT serialized_model FROM {model_table} WHERE model_name = %s;"
cursor = connection.cursor()
cursor.execute(query, (model_name,))
model_serialized = cursor.fetchone()[0]

# Deserialize the model
model_config = tf.keras.models.model_from_json(model_serialized)
loaded_model = tf.keras.models.Sequential()

for layer_config in model_config['config']['layers']:
    layer = tf.keras.layers.deserialize(layer_config, custom_objects={})
    loaded_model.add(layer)



# Execute a query to fetch data
query_data = "SELECT * FROM stock_data_AAL"
df = pd.read_sql(query_data, connection)


# Convert the 'date' column to datetime with the correct format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
close_column = df['close']
df.dropna(inplace=True)
close_column = df['close']
df.dropna(inplace=True)
df.dropna(inplace=True)
# Calculate the 100-day and 200-day moving averages
ma_100_days = close_column.rolling(window=100).mean()
ma_200_days = close_column.rolling(window=200).mean()

# Create a Tkinter window
root = tk.Tk()
root.title("Close Value Lookup")

# Function to fetch and display the close value for the entered date
def fetch_close_value():
    date_str = date_entry.get()
    entered_date = pd.to_datetime(date_str, errors='coerce')

    if not pd.isnull(entered_date):
        try:
            close_value = close_column[df['date'] == entered_date].iloc[0]
            close_label.config(text=f"Close Value on {entered_date.strftime('%Y-%m-%d')}: {close_value:.2f}")
        except IndexError:
            close_label.config(text=f"No data for {entered_date.strftime('%Y-%m-%d')}")
    else:
        close_label.config(text="Invalid date format")

# Create Tkinter widgets
date_label = Label(root, text="Enter Date (MM/DD/YYYY):")
date_entry = Entry(root)
fetch_button = Button(root, text="Fetch Close Value", command=fetch_close_value)
close_label = Label(root, text="")

# Pack Tkinter widgets
date_label.pack()
date_entry.pack()
fetch_button.pack()
close_label.pack()

# Function to plot the data
def plot_data():
    plt.clf()  # Clear the previous plot
    plt.plot(ma_100_days, 'r', label='MA 100 days')
    plt.plot(ma_200_days, 'b', label='MA 200 days')
    plt.plot(close_column, 'g', label='Close Price')
    plt.legend()
    plt.grid(True)
    canvas.draw()

# Create a Matplotlib figure
figure, ax = plt.subplots(figsize=(8, 6))

# Matplotlib Plotting within Tkinter window
plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.BOTTOM, pady=10)

canvas = FigureCanvasTkAgg(figure, master=plot_frame)
canvas.get_tk_widget().pack()

# Plot the initial data
plot_data()

# Start Tkinter event loop
root.mainloop()

# The rest of your code...
