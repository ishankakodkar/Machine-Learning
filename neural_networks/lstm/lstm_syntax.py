# This file demonstrates the syntax for building a Long Short-Term Memory (LSTM) network
# for sequence prediction using the TensorFlow and Keras libraries.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Create sample sequence data
# Input shape for an LSTM is (n_samples, n_timesteps, n_features).
# Let's create a simple dataset to predict the next value in a sine wave.
n_samples = 100
n_timesteps = 10
n_features = 1

# Generate sine wave data
# We will create sequences of 10 points and use them to predict the 11th point.
X = []
y = []
for i in range(n_samples + n_timesteps):
    X.append(np.sin(np.linspace(i, i + n_timesteps, n_timesteps)))
    y.append(np.sin(i + n_timesteps))

X = np.array(X[:-n_timesteps]).reshape(n_samples, n_timesteps, n_features)
y = np.array(y[:-n_timesteps])

# 2. Build the LSTM model
# - LSTM: The Long Short-Term Memory layer.
#   - units: The dimensionality of the output space (number of neurons in the layer).
#   - activation: Activation function to use ('tanh' is the default).
#   - return_sequences (bool): Whether to return the last output in the output sequence,
#     or the full sequence. This must be True for all but the last LSTM layer if you are stacking them.
#   - input_shape: (n_timesteps, n_features) for the first layer.

model = Sequential([
    # First LSTM layer
    # `return_sequences=True` is needed to pass the full sequence to the next LSTM layer.
    LSTM(units=50, return_sequences=True, input_shape=(n_timesteps, n_features)),
    
    # Second LSTM layer
    # `return_sequences=False` (the default) because we only need the final output
    # to pass to the Dense layer.
    LSTM(units=50),
    
    # Dense output layer
    Dense(units=1) # For regression
])

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

# 4. Inspect the model
print("--- Model Architecture ---")
model.summary()

# 5. Train the model
print("\n--- Training Model ---")
model.fit(X, y, epochs=30, batch_size=16, verbose=0)
print("Model training complete.")

# 6. Make predictions
# Create a new sample with the correct shape (1, n_timesteps, n_features)
new_sequence = np.sin(np.linspace(100, 100 + n_timesteps, n_timesteps))
new_X = new_sequence.reshape(1, n_timesteps, n_features)

prediction = model.predict(new_X)
actual = np.sin(100 + n_timesteps)

print("\n--- Making a Prediction ---")
print(f"Input sequence shape: {new_X.shape}")
print(f"Predicted value: {prediction[0][0]:.4f}")
print(f"Actual value:    {actual:.4f}")
