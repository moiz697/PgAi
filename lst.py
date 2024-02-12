import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label, Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import psycopg2
import io
from tensorflow.keras.models import save_model
import tempfile
# Load environment variables from .env file
from sklearn.metrics import mean_squared_error
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
def save_model_to_db(model, model_name, connection):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Save the model to the temporary file
        model_file_path = tmp_file.name
        model.save(model_file_path, save_format='h5')  # Explicitly specify HDF5 format
        
    # Read the temporary file into a BytesIO object
    with open(model_file_path, 'rb') as model_file:
        model_byte_array = model_file.read()
    
    # Insert the model data into the database
    with connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO keras_models (model_name, model_data)
            VALUES (%s, %s)
            ON CONFLICT (model_name) DO UPDATE
            SET model_data = EXCLUDED.model_data
        """, (model_name, psycopg2.Binary(model_byte_array)))
        connection.commit()
    
    # Clean up the temporary file
    os.remove(model_file_path)

def load_model_from_db(model_name, connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT model_data FROM keras_models WHERE model_name = %s", (model_name,))
        result = cursor.fetchone()
        if result:
            model_byte_array = result[0]

            # Testing code starts here
            temp_path = "temp_model.h5"
            with open(temp_path, "wb") as f:
                f.write(model_byte_array)

            import h5py
            try:
                with h5py.File(temp_path, 'r') as hf:
                    print("Success! The file is a valid HDF5 file.")
                    # If success, proceed to load the model as usual
                    model_buffer = io.BytesIO(model_byte_array)
                    model = load_model(model_buffer)
                    return model
            except OSError:
                print("Failure. The file is not a valid HDF5 file.")
                return None
            # Testing code ends here

        else:
            return None


# Execute a query to fetch data
query = "SELECT * FROM PGDATA"
df = pd.read_sql(query, connection)

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

data_train = df['close'][0: int(len(df)*0.60)]
data_test = df['close'][int(len(df)*0.60):] 
print(data_train)
print(data_test)


data_train.shape[0]
data_test.shape[0]

scaler=MinMaxScaler(feature_range=(0,1))
# Reshape data_train and data_test before scaling
data_train_scale = scaler.fit_transform(data_train.values.reshape(-1, 1))
data_test_scale = scaler.transform(data_test.values.reshape(-1, 1))

# Prepare the training data
x = []
y = []
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i, 0])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)

# Reshape x to be a 3D array
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Create the LSTM model



def create_lstm_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))  # Increased dropout rate
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))  # Increased dropout rate
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))  # Increased dropout rate
    
    
    model.add(Dense(units=1))
    
    return model

    

# Reshape x to be a 3D array
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Create the LSTM model
model = create_lstm_model(input_shape=(x.shape[1], 1))


optimizer = Nadam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mean_squared_error')


# Define callbacks
callbacks = [
    ModelCheckpoint('Save.keras', save_best_only=True),
    TensorBoard(log_dir='./logs', histogram_freq=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]
save_model_to_db(model, 'MyKerasModel', connection)
# Training the model
history = model.fit(x, y, epochs=50, batch_size=64, verbose=1, validation_split=0.5, callbacks=callbacks)


# Save the model after training to PostgreSQL
save_model_to_db(model, 'MyKerasModel', connection)

# No need to call model.save('Save.keras') if you're saving to PostgreSQL

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model after training
model.save('Save.keras')


# Load the saved model using Keras
loaded_model = load_model('Save.keras')
# Use the same MinMaxScaler to scale the test data
data_test_scale = scaler.transform(data_test.values.reshape(-1, 1))

# Prepare the test data
x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i, 0])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape x_test to be a 3D array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the loaded model
predictions = loaded_model.predict(x_test)

# Inverse transform the predictions to get the original scale
predicted_values = scaler.inverse_transform(predictions.reshape(-1, 1))

# Evaluate your model or do further analysis with the predictions and actual values
# ...

# Print the first few predictions for visualization
print("Predicted Values:")
print(predicted_values[10868:])
print("Actual Values:")
print(data_test.values[10868:10868])



model.summary()

pass_100_days=df.tail(100)
data_test=pd.concat([pass_100_days,data_test],ignore_index=True)
print(data_test)
loaded_model = load_model_from_db('MyKerasModel', connection)
# Inverse transform the predictions to get them back to the original scale if you used MinMaxScaler
predicted_values = scaler.inverse_transform(predictions)

# Compare predicted values to the actual values (y_test)
# Assuming you have an actual y_test that corresponds to x_test
actual_values = y_test  # Replace with actual y_test data
actual_values_scaled = scaler.transform(actual_values.reshape(-1, 1))
actual_values_inverse = scaler.inverse_transform(actual_values_scaled)



connection.close()