import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# 1. Load and prepare the data
dataset = pd.read_csv('../cnn/historical_prices.csv') # Re-using the same dataset
prices = dataset[['Close']].values

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# 2. Create training and test sets
SEQUENCE_LENGTH = 10
X, y = create_sequences(prices_scaled, SEQUENCE_LENGTH)

# Split data into training and testing: 80% training, 20% testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    LSTM(units=50),
    Dense(units=1)
])

# 4. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the model
print("Training the LSTM model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 6. Make predictions
print("\nMaking predictions...")
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)

# 7. Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
