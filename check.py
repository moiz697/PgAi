import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import psycopg2

# Load environment variables from .env file
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

# Fetch the latest model from the ml_models table
cursor = connection.cursor()
cursor.execute("SELECT serialized_model FROM ml_models ORDER BY model_id DESC LIMIT 1;")
serialized_model = cursor.fetchone()[0]
cursor.close()

# Load the serialized model using model_from_json
loaded_model = model_from_json(serialized_model)

# Use the same MinMaxScaler to scale the data for predictions
scaler = MinMaxScaler(feature_range=(0, 1))  # Ensure you use the same scaler parameters as before

# Fetch the test data or data for prediction from the database
cursor = connection.cursor()
cursor.execute("SELECT * FROM stock_data WHERE date >= '01/01/2019';")  # Replace 'your_start_date' with the desired start date
data_for_prediction = cursor.fetchall()
cursor.close()

# Convert the data_for_prediction to a Pandas DataFrame
df_for_prediction = pd.DataFrame(data_for_prediction, columns=['date', 'close'])  # Adjust column names as needed

# Convert the 'date' column to datetime with the correct format
df_for_prediction['date'] = pd.to_datetime(df_for_prediction['date'], format='%d/%m/%Y')

# Calculate the 100-day and 200-day moving averages
ma_100_days_for_prediction = df_for_prediction['close'].rolling(window=100).mean()
ma_200_days_for_prediction = df_for_prediction['close'].rolling(window=200).mean()

# Preprocess the data for prediction
# Assuming you have already fetched data_for_prediction_scale as mentioned in the previous code

# Reshape data_for_prediction_scale before using it in prediction
data_for_prediction_scale = data_for_prediction_scale.reshape(-1, 1)

# Use the same MinMaxScaler to transform the data
data_for_prediction_transformed = scaler.transform(data_for_prediction_scale)

# Prepare the data for prediction
x_pred = []
for i in range(100, data_for_prediction_transformed.shape[0]):
    x_pred.append(data_for_prediction_transformed[i-100:i, 0])

x_pred = np.array(x_pred)
x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='mean_squared_error')

# Make predictions using the loaded model
predictions = loaded_model.predict(x_pred)

# Inverse transform the predictions to get the original scale
predicted_values = scaler.inverse_transform(predictions.reshape(-1, 1))

# Print or use the predicted values as needed
print("Predicted Values:")
print(predicted_values)

# Close the database connection
connection.close()
