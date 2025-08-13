# This file demonstrates the syntax for building an Encoder-Decoder (Seq2Seq) model
# for multi-step time series forecasting using TensorFlow and Keras.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# 1. Create sample sequence data
# The task is to predict an output sequence from an input sequence.
# Input shape: (n_samples, n_timesteps_in, n_features)
# Output shape: (n_samples, n_timesteps_out, n_features)

n_samples = 100
n_timesteps_in = 10
n_timesteps_out = 5
n_features = 1

# Generate data where the output is a simple transformation of the input
X = np.random.rand(n_samples, n_timesteps_in, n_features)
y = np.zeros((n_samples, n_timesteps_out, n_features))
for i in range(n_samples):
    # Example target: the next 5 values are the last input value repeated and scaled
    last_val = X[i, -1, 0]
    y[i, :, 0] = [last_val * (1 + j*0.1) for j in range(n_timesteps_out)]

# 2. Build the Encoder-Decoder model
# - The Encoder is an LSTM that reads the input sequence and outputs a context vector (its final state).
# - RepeatVector repeats this context vector for each step of the output sequence.
# - The Decoder is another LSTM that takes the context vector and generates the output sequence.
# - TimeDistributed applies a Dense layer to each time step of the decoder's output.

model = Sequential([
    # --- Encoder ---
    # The encoder LSTM processes the input sequence. We only care about its final state.
    LSTM(units=100, activation='relu', input_shape=(n_timesteps_in, n_features)),
    
    # --- Context Vector Repeater ---
    # The context vector (encoder's final state) is repeated n_timesteps_out times.
    # This provides the input for each step of the decoder.
    RepeatVector(n_timesteps_out),
    
    # --- Decoder ---
    # The decoder LSTM generates the output sequence. `return_sequences=True` is crucial.
    LSTM(units=100, activation='relu', return_sequences=True),
    
    # --- Output Layer ---
    # A TimeDistributed Dense layer applies the same dense operation to every timestep
    # of the decoder's output, producing a prediction at each output step.
    TimeDistributed(Dense(units=n_features))
])

# 3. Compile the model
model.compile(optimizer='adam', loss='mse')

# 4. Inspect the model
print("--- Model Architecture ---")
model.summary()

# 5. Train the model
print("\n--- Training Model ---")
model.fit(X, y, epochs=50, batch_size=16, verbose=0)
print("Model training complete.")

# 6. Make predictions
# Create a new sample with the correct shape (1, n_timesteps_in, n_features)
new_X = np.random.rand(1, n_timesteps_in, n_features)
prediction = model.predict(new_X)

print("\n--- Making a Prediction ---")
print(f"Input sequence shape: {new_X.shape}")
print(f"Predicted output sequence shape: {prediction.shape}")
print(f"Predicted sequence:\n{prediction[0].flatten()}")
