import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array
# Replace 'your_file.csv' with the actual file path
df = pd.read_csv('/Users/moizibrar/Downloads/AAPL.csv')

# Display the last few rows of the DataFrame to understand the structure
print(df.tail())

# Reset the index and extract the 'close' column
df1 = df.reset_index()['close']

# Plot the 'close' column
plt.plot(df1)

# Display the values of the 'close' column
print(df1)

# Normalize the 'close' values using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Display the shape of the normalized data
print(df1.shape)

# Define the size of the training set
train_size = int(len(df1) * 0.65)
test_size = len(df1) - train_size

# Create training and testing datasets
train_data = df1[:train_size, :]
test_data = df1[train_size:, :1]

# Display the sizes of the training and testing datasets
print(train_size)
print(test_size)
print(test_data)

def create_dataset(dataset, time_step=1):
    """
    Create sequences of input features (dataX) and corresponding target values (dataY)
    from a given time series dataset.

    Parameters:
    - dataset: 1D numpy array representing the time series data.
    - time_step: Number of time steps to consider for each sequence (default is 1).

    Returns:
    - dataX: 2D numpy array representing input sequences.
    - dataY: 1D numpy array representing target values.
    """
    dataX, dataY = [], []

    # Iterate over the dataset to create sequences
    for i in range(len(dataset) - time_step - 1):
        # Extract a sequence of 'time_step' elements from the dataset
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)

        # The target value is the next element after the sequence
        dataY.append(dataset[i + time_step, 0])

    # Convert the lists to numpy arrays
    return np.array(dataX), np.array(dataY)

# Define the number of time steps for each sequence
time_step = 100

# Create training datasets using the create_dataset function
x_train, y_train = create_dataset(train_data, time_step)

# Create testing datasets using the create_dataset function
x_test, y_test = create_dataset(test_data, time_step)

# Print the shapes of the training and testing datasets
print("Training data shapes:")
print("Input sequences (x_train):", x_train.shape)
print("Target values (y_train):", y_train.shape)

print("\nTesting data shapes:")
print("Input sequences (x_test):", x_test.shape)
print("Target values (y_test):", y_test.shape)

# Reshape x_train to be compatible with the input shape of a neural network (3D array)
# The new shape is (number of samples, number of time steps, number of features per time step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# Reshape x_test similarly to match the input shape of the neural network
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Create a sequential model
model = Sequential()

# Add the first LSTM layer with 50 units, return full sequences, and specify input shape
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))

# Add the second LSTM layer with 50 units and return full sequences
model.add(LSTM(50, return_sequences=True))

# Add the third LSTM layer with 50 units (final layer in the sequence)
model.add(LSTM(50))

# Add a dense output layer with 1 unit for regression task
model.add(Dense(1))

# Compile the model, specifying mean squared error loss and Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Display the model summary
model.summary()

# Train the model on the training data with validation on the testing data
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, verbose=1)

# Check the TensorFlow version
print(tf.__version__)

# Predict on the training and testing datasets
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Inverse transform the predictions to the original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate and display the Root Mean Squared Error (RMSE) for the training data
rmse_train = math.sqrt(mean_squared_error(y_train, train_predict))
print("Root Mean Squared Error (RMSE) for Training Data:", rmse_train)

look_back=100
trainPredictPlot=np.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=np.empty_like(df1)
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
x_input=test_data[341:].reshape(1,-1)
x_input.shape
len(test_data)
x_input=test_data[341:].reshape(1,-1)
x_input.shape
temp_input = list(x_test)
temp_input = temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
temp_input = list(x_test)
temp_input = temp_input[0].tolist()
lst_output = []
n_steps = 100
i = 0
while i < 30:
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[-n_steps:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input = np.append(temp_input, yhat[0, 0])
        lst_output = np.append(lst_output, yhat[0, 0])
        i = i + 1
    else:
        x_input = np.array(temp_input[-n_steps:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat[0]))
        temp_input = np.append(temp_input, yhat[0, 0])
        lst_output = np.append(lst_output, yhat[0, 0])
        i = i + 1
temp_input = np.array(temp_input)
lst_output = np.array(lst_output)   

print(lst_output)

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)
len(df1)
plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
df3 = scaler.inverse_transform(df3).tolist()
plt.plot(df3)
plt.show()
