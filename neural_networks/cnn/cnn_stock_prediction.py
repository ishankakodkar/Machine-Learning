import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Function to create sequences from time-series data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # The sequence of prices will be our features
        X.append(data[i:(i + sequence_length), 0])
        # The direction of the next price is our target
        y.append(1 if data[i + sequence_length, 0] > data[i + sequence_length - 1, 0] else 0)
    return np.array(X), np.array(y)

# 1. Load the dataset
dataset = pd.read_csv('historical_prices.csv')
prices = dataset[['Close']].values

# 2. Scale the data
scaler = StandardScaler()
prices_scaled = scaler.fit_transform(prices)

# 3. Create sequences
SEQUENCE_LENGTH = 10
X, y = create_sequences(prices_scaled, SEQUENCE_LENGTH)

# Reshape X for CNN input [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=50, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model
print("Training the CNN model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 8. Evaluate the model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')
