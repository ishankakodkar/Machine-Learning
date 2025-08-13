# This file demonstrates the syntax for building a 1D Convolutional Neural Network (CNN)
# for sequence data using the TensorFlow and Keras libraries.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 1. Create sample sequence data
# A common task is to predict a future value based on a sequence of past values.
# Input shape for a 1D CNN is (n_samples, n_timesteps, n_features).
# Let's create 100 samples, each with 10 timesteps and 1 feature.

n_samples = 100
n_timesteps = 10
n_features = 1

X = np.random.rand(n_samples, n_timesteps, n_features)
# Create a simple target: the sum of the last 3 values in the sequence
y = np.sum(X[:, -3:, :], axis=1)

# 2. Build the 1D CNN model
# - Conv1D: 1D convolution layer.
#   - filters: The number of output filters in the convolution (i.e., number of features to learn).
#   - kernel_size: The length of the 1D convolution window.
#   - activation: Activation function ('relu' is common).
#   - input_shape: (n_timesteps, n_features) for the first layer.
# - MaxPooling1D: Downsamples the input along its temporal dimension.
#   - pool_size: The size of the max pooling window.
# - Flatten: Flattens the output of the convolutional layers to be fed into Dense layers.

model = Sequential([
    # Convolutional base
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    
    # Flatten the feature maps to feed into the dense layers
    Flatten(),
    
    # Dense classifier/regressor part
    Dense(units=50, activation='relu'),
    Dense(units=1) # Output layer for regression
])

# 3. Compile the model
# For regression, we use a loss like Mean Squared Error.
model.compile(
    optimizer='adam',
    loss='mse'
)

# 4. Inspect the model
print("--- Model Architecture ---")
model.summary()

# 5. Train the model
print("\n--- Training Model ---")
model.fit(X, y, epochs=50, batch_size=16, verbose=0)
print("Model training complete.")

# 6. Make predictions
# Create a new sample with the correct shape (1, n_timesteps, n_features)
new_X = np.random.rand(1, n_timesteps, n_features)
prediction = model.predict(new_X)

print("\n--- Making a Prediction ---")
print(f"Input sequence shape: {new_X.shape}")
print(f"Predicted value: {prediction[0][0]:.4f}")
