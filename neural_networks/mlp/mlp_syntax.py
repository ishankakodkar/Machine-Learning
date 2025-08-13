# This file demonstrates the syntax for building a Multilayer Perceptron (MLP)
# using the TensorFlow and Keras libraries.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# 1. Create sample data
# Let's create some data for a simple regression task.
# y = 3*x1 + 2*x2 + 5
X = np.random.rand(100, 2) * 10 # 100 samples, 2 features
y = 3 * X[:, 0] + 2 * X[:, 1] + 5 + np.random.randn(100) # Add some noise

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Build the MLP model
# We use the Sequential API, which allows us to build a model layer by layer.
# - Dense: A standard fully connected neural network layer.
#   - units: The number of neurons in the layer.
#   - activation: The activation function to use ('relu', 'sigmoid', 'linear', etc.).
#   - input_shape: Required for the first layer to know the shape of the input data.

model = Sequential([
    # Input layer and first hidden layer
    # Takes input with 2 features, has 32 neurons, uses ReLU activation
    Dense(units=32, activation='relu', input_shape=(2,)),
    
    # Second hidden layer
    Dense(units=16, activation='relu'),
    
    # Output layer
    # Has 1 neuron because we are predicting a single continuous value (regression)
    # Uses a 'linear' activation function for regression tasks
    Dense(units=1)
])

# 3. Compile the model
# Before training, we must configure the learning process using .compile().
# - optimizer: The algorithm to use to update the weights (e.g., 'adam', 'sgd', 'rmsprop').
#   Adam is a popular and effective default choice.
# - loss: The function to measure how inaccurate the model is during training.
#   'mean_squared_error' (mse) is standard for regression.
#   'binary_crossentropy' for binary classification.
# - metrics: A list of metrics to be evaluated by the model during training and testing.

model.compile(
    optimizer='adam',
    loss='mean_squared_error', # or 'mse'
    metrics=['mae'] # Mean Absolute Error
)

# 4. Inspect the model
# .summary() prints a string summary of the network.
print("--- Model Architecture ---")
model.summary()

# 5. Train the model
# The .fit() method trains the model for a fixed number of epochs (iterations on a dataset).
# - epochs: Number of times to iterate over the entire dataset.
# - batch_size: Number of samples per gradient update.
# - validation_split: Fraction of the training data to be used as validation data.

print("\n--- Training Model ---")
history = model.fit(
    X_scaled,
    y,
    epochs=50, # Train for 50 passes over the data
    batch_size=10,
    validation_split=0.2, # Use 20% of data for validation
    verbose=0 # Set to 1 to see training progress
)
print("Model training complete.")

# 6. Make predictions
# New data must be scaled with the same scaler.
new_X = np.array([[5, 5], [10, 2]])
new_X_scaled = scaler.transform(new_X)
predictions = model.predict(new_X_scaled)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"Prediction for {val}: {predictions[i][0]:.2f}")
