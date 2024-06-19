import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Load the data from the CSV file
file_path = 'TSLA.csv'  # Make sure this path is correct
data = pd.read_csv(file_path)

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])  # Assuming the date column is named 'Date'
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Feature engineering
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['DayOfWeek'] = data.index.dayofweek

# Adding rolling averages and other features
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Volatility'] = data['Close'].rolling(window=10).std()

# Drop rows with NaN values created by rolling features
data.dropna(inplace=True)

# Define the features and target
X = data[['Year', 'Month', 'Day', 'DayOfWeek', 'MA10', 'MA50', 'MA200', 'Volatility']]
y = data['Close'].values  # Assuming the close price column is named 'Close'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': stats.randint(50, 200),
    'learning_rate': stats.uniform(0.01, 0.2),
    'max_depth': stats.randint(3, 6)
}

random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train_scaled, y_train)

# Best model
best_model = random_search.best_estimator_
print(f'Best Parameters: {random_search.best_params_}')

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation MSE: {-np.mean(cv_scores)}')

# Track training and validation error for each iteration
train_errors = []
test_errors = []

for y_train_pred in best_model.staged_predict(X_train_scaled):
    train_errors.append(mean_squared_error(y_train, y_train_pred))

for y_test_pred in best_model.staged_predict(X_test_scaled):
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Plot the training and validation error over the iterations
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(train_errors)) + 1, train_errors, label='Training Error')
plt.plot(np.arange(len(test_errors)) + 1, test_errors, label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Error Over Iterations')
plt.legend()
plt.show()

# Train the best model on the entire training set
best_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model and scaler in PostgreSQL
def save_model_and_scaler_to_postgres(model, scaler, model_name):
    model_data = pickle.dumps(model)
    scaler_data = pickle.dumps(scaler)

    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    insert_query = sql.SQL("""
        INSERT INTO tesla_model_storage (model_name, model_data, scaler_data)
        VALUES (%s, %s, %s)
        ON CONFLICT (model_name)
        DO UPDATE SET model_data = EXCLUDED.model_data, scaler_data = EXCLUDED.scaler_data;
    """)
    cursor.execute(insert_query, (model_name, psycopg2.Binary(model_data), psycopg2.Binary(scaler_data)))
    conn.commit()
    cursor.close()
    conn.close()

save_model_and_scaler_to_postgres(best_model, scaler, 'tesla_gradient_boosting_model')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Actual Close Prices')

# Predict the entire data range for a smoother plot
full_data_scaled = scaler.transform(X)
full_predictions = best_model.predict(full_data_scaled)
plt.plot(data.index, full_predictions, label='Predicted Close Prices', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.legend()
plt.show()
