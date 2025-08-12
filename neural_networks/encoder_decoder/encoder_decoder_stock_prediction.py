import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

# Function to create sequences for seq2seq model
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 1. Load and prepare the data
dataset = pd.read_csv('../cnn/historical_prices.csv') # Re-using the same dataset
prices = dataset[['Close']].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# 2. Define sequence lengths and create sequences
N_STEPS_IN = 10  # Length of input sequence
N_STEPS_OUT = 5   # Length of output sequence
X, y = create_sequences(prices_scaled, N_STEPS_IN, N_STEPS_OUT)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))

# 3. Split data into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Build the Encoder-Decoder model
model = Sequential([
    LSTM(100, activation='relu', input_shape=(N_STEPS_IN, n_features)), # Encoder
    RepeatVector(N_STEPS_OUT),                                         # Bridge
    LSTM(100, activation='relu', return_sequences=True),               # Decoder
    TimeDistributed(Dense(1))                                          # Output layer
])

# 5. Compile the model
model.compile(optimizer='adam', loss='mse')

# 6. Train the model
print("Training the Encoder-Decoder model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 7. Make a prediction
print("\nMaking a prediction on a sample from the test set...")
x_input = X_test[0].reshape((1, N_STEPS_IN, n_features))
yhat = model.predict(x_input, verbose=0)

# 8. Visualize the prediction
predicted_sequence = scaler.inverse_transform(yhat[0])
actual_sequence = scaler.inverse_transform(y_test[0])

print(f"Predicted Sequence: {predicted_sequence.flatten()}")
print(f"Actual Sequence:    {actual_sequence.flatten()}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, N_STEPS_OUT + 1), actual_sequence, marker='o', label='Actual')
plt.plot(range(1, N_STEPS_OUT + 1), predicted_sequence, marker='o', label='Predicted')
plt.title('Multi-Step Stock Price Prediction')
plt.xlabel('Future Timestep')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
