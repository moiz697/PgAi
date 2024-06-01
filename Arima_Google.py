import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from datetime import date, timedelta

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

# Define the file path
file_path = '/Users/umerkhurshid/Downloads/GOOG.csv'
df = pd.read_csv(file_path)

# Select necessary columns
df = df[['Date', 'Close']]

# Convert the 'Date' column to datetime with dayfirst=True
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Drop rows with NaT in the 'Date' column
df = df.dropna(subset=['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Ensure 'Close' column is numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop rows with NaN values in the 'Close' column
df = df.dropna(subset=['Close'])

# Ensure there are no infinite values in the 'Close' column
df = df[np.isfinite(df['Close'])]

# Plot the data
#fig = px.line(df, x=df.index, y='Close', title='Google Stock Data')
#fig.show()

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistics: %f' % result[0])
    print('P-Value: %f' % result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] <= 0.05:
        print("Reject the Null Hypothesis: Data is stationary")
    else:
        print("Fail to reject the Null Hypothesis: Data is not stationary")

# Check the stationarity of the 'Close' prices
check_stationarity(df['Close'])

# Decompose the data for trends, seasonality, and noise
decompose = seasonal_decompose(df['Close'], model='additive', period=30)
#decompose.plot()
#plt.show()

# Step 3: Finding the value of D for ARIMA
# Original Series 
#fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
#axes[0, 0].plot(df['Close'])
#axes[0, 0].set_title('Original Series')
#plot_acf(df['Close'], ax=axes[0, 1])

# 1st Differencing
#axes[1, 0].plot(df['Close'].diff())
#axes[1, 0].set_title('1st Order Differencing')
#plot_acf(df['Close'].diff().dropna(), ax=axes[1, 1])

# 2nd Order Differencing
#axes[2, 0].plot(df['Close'].diff().diff())
#axes[2, 0].set_title('2nd Order Differencing')
#plot_acf(df['Close'].diff().diff().dropna(), ax=axes[2, 1])

#plt.show()

# Value of D=1, now we have to find P Value which is Auto-Regressive Value
#pd.plotting.autocorrelation_plot(df['Close'])
#plt.show()

#plot_acf(df['Close'], alpha=0.05)
#plt.show()

x_acf = pd.DataFrame(acf(df['Close']))
#print(x_acf)

# After 2 Values lag is going down by 95% so P=2

# Now finding the Q value
#plot_pacf(df['Close'], lags=20, alpha=0.05)
#plt.show()

# Model parameters
p = 2
d = 1
q = 2
P = 1
D = 1
Q = 2
s = 12


import statsmodels.api as sm
import warnings
# Fit SARIMAX model
model = sm.tsa.statespace.SARIMAX(df['Close'], order=(p, d, q), seasonal_order=(P, D, Q, s))
fit_model = model.fit()
print(fit_model.summary())

# Forecasting
forecast_periods = 20
forecast = fit_model.get_forecast(steps=forecast_periods)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Get the date range for plotting
start_date = df.index[-1] + pd.DateOffset(days=1)  # Start one day after the last date in the dataset
forecast_dates = pd.date_range(start=start_date, periods=forecast_periods, freq='D')

# Plotting historical data and forecasts
plt.figure(figsize=(14,7))
plt.plot(df.index, df['Close'], label='Actual')
plt.plot(forecast_dates, forecast_mean.values, color='red', label='Forecast')
plt.fill_between(forecast_dates, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='Confidence Intervals')

plt.title('Google Stock Closing Prices - Forecast for Next 15 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plotting only the trend (without actual and forecast)
plt.figure(figsize=(14,7))
plt.plot(df.index, df['Close'], label='Actual', alpha=0.5)
plt.plot(forecast_dates, forecast_mean.values, color='red', label='Forecast')


specific_date = pd.Timestamp('30-05-2024')  # Change this date to your desired date

# Check if the specific date is within the forecast_dates range
if specific_date in forecast_dates:
    index = forecast_dates.get_loc(specific_date)
    forecasted_close = forecast_mean[index]
    print(f"Forecasted close price for {specific_date.date()} is: {forecasted_close:.2f}")
else:
    print(f"No forecast available for {specific_date.date()}.")
