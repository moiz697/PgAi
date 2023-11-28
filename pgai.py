import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import math




import os
import psycopg2

def connect_to_db():
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

    return connection

# Call the function to connect to the database
connection = connect_to_db()

def load_and_prepare_data(connection):
    # Get database connection details from the connection object
    db_user = connection.info.user
    db_password = connection.info.password
    db_host = connection.info.host
    db_port = connection.info.port
    db_name = connection.info.dbname

    # Create a new connection using SQLAlchemy
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    # Rest of the function remains unchanged
    sql_query = "SELECT * FROM stock_data"
    df = pd.read_sql_query(sql_query, engine)
    df1 = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    train_size = int(len(df1) * 0.70)
    test_size = len(df1) - train_size
    train_data = df1[:train_size, :]
    test_data = df1[train_size:, :1]
    engine.dispose()
    return df1, train_data, test_data, scaler

def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def create_and_train_model(x_train, y_train, x_test, y_test):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(100, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, verbose=1)
    return model

def predict_and_evaluate(model, x_train, y_train, x_test, y_test, scaler):
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    rmse_train = math.sqrt(mean_squared_error(y_train, train_predict))
    return train_predict, test_predict, rmse_train

def plot_original_vs_predicted(df1, train_predict, test_predict, scaler):
    # Inverse transform the original normalized data
    original_data = scaler.inverse_transform(df1)

    # Prepare the plot for the training predictions
    train_predict_plot = np.empty_like(df1)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[100:len(train_predict) + 100, :] = train_predict

    # Prepare the plot for the testing predictions
    test_predict_plot = np.empty_like(df1)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (100 * 2) + 1:len(df1) - 1, :] = test_predict

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title('Original vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.plot(original_data, label='Original Data')
    plt.plot(train_predict_plot, label='Training Predictions')
    plt.plot(test_predict_plot, label='Testing Predictions')
    plt.legend()
    plt.show()

    print("Plotting Original vs Predicted Stock Prices")
    print("Blue line represents the original stock prices.")
    print("Orange line represents the training data predictions.")
    print("Green line represents the testing data predictions.")

def plot_future_predictions(future_predictions):
    if len(future_predictions) == 0:
        print("No future predictions generated.")
        return

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title('Future Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.plot(future_predictions, label='Future Predictions', color='red')
    plt.legend()
    plt.show()

    print("Plotting Future Stock Price Predictions")
    print("Red line represents the future predictions for the stock prices.")


def plot_original_vs_predicted(df1, train_predict, test_predict, scaler):
    # Inverse transform the original normalized data
    original_data = scaler.inverse_transform(df1)

    # Prepare the plot for the training predictions
    train_predict_plot = np.empty_like(df1)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[100:len(train_predict) + 100, :] = train_predict

    # Prepare the plot for the testing predictions
    test_predict_plot = np.empty_like(df1)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (100 * 2) + 1:len(df1) - 1, :] = test_predict

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title('Original vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.plot(original_data, label='Original Data')
    plt.plot(train_predict_plot, label='Training Predictions')
    plt.plot(test_predict_plot, label='Testing Predictions')
    plt.legend()
    plt.show()

    print("Plotting Original vs Predicted Stock Prices")
    print("Blue line represents the original stock prices.")
    print("Orange line represents the training data predictions.")
    print("Green line represents the testing data predictions.")

def plot_future_prediction(df1, future_predictions, scaler):
    # Inverse transform the future predictions to the original scale
    future_predictions_transformed = scaler.inverse_transform(future_predictions.reshape(-1, 1))

    # Define the range for the future predictions
    last_data_point = len(df1)
    future_range = np.arange(last_data_point, last_data_point + len(future_predictions))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title('Future Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.plot(future_range, future_predictions_transformed, label='Future Predictions', color='red')
    plt.legend()
    plt.show()

    print("Plotting Future Stock Price Predictions")
    print("Red line represents the future predictions for the stock prices.")

    return future_predictions_transformed  # Add this line to return the transformed predictions


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def apply_kfold_cross_validation(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_values = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = calculate_rmse(y_test, y_pred)
        rmse_values.append(rmse)

    return rmse_values

connection = connect_to_db()
if connection:
    df1, train_data, test_data, scaler = load_and_prepare_data(connection)
    x_train, y_train = create_dataset(train_data)
    x_test, y_test = create_dataset(test_data)
    model = create_and_train_model(x_train, y_train, x_test, y_test)
    train_predict, test_predict, rmse_train = predict_and_evaluate(model, x_train, y_train, x_test, y_test, scaler)
    lst_output = plot_future_prediction(df1, test_predict, scaler)
    plot_future_prediction(df1, lst_output, scaler)
    rmse_train = calculate_rmse(y_train, train_predict)
    rmse_test = calculate_rmse(y_test, test_predict)

    # Print RMSE values
    print("RMSE for Training Data:", rmse_train)
    print("RMSE for Testing Data:", rmse_test)
    # Apply K-fold cross-validation on the training data
    rmse_values = apply_kfold_cross_validation(model, x_train, y_train)

    # Print average RMSE over all folds
    print("Average RMSE over K folds:", np.mean(rmse_values))


    train_predict, test_predict = predict_and_evaluate(model, x_train, y_train, x_test, y_test, scaler)[:2]
    
    plot_original_vs_predicted(df1, train_predict, test_predict, scaler)
    plot_future_predictions(lst_output)
    connection.close()
