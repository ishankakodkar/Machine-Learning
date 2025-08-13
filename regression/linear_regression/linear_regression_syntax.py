# This file demonstrates the syntax for using the LinearRegression model from scikit-learn.
# It focuses on the class itself, its parameters, and how to use it.

import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Create sample data
# Let's create a simple dataset where y = 2*x + 1
X = np.array([[1], [2], [3], [4], [5]])  # Feature matrix (needs to be 2D)
y = np.array([3, 5, 7, 9, 11])           # Target vector

# 2. Instantiate the LinearRegression model
# The LinearRegression class has several parameters you can tweak:
# - fit_intercept (bool, default=True): Whether to calculate the intercept (beta_0).
#   If set to False, no intercept will be used in calculations (i.e., the line will pass through the origin).
# - normalize (bool, default=False): This parameter is deprecated in favor of using
#   sklearn.preprocessing.StandardScaler before fitting the model.
# - n_jobs (int, default=None): The number of CPU cores to use for computation.
#   -1 means using all available cores.

model = LinearRegression(
    fit_intercept=True
)

# 3. Train the model
# The .fit() method trains the model on the provided data.
# It learns the optimal values for the coefficients (beta_1) and the intercept (beta_0).
model.fit(X, y)

# 4. Inspect the learned parameters
# After fitting, the model stores the learned parameters in its attributes:
# - .coef_: Contains the coefficient(s) for each feature (beta_1).
# - .intercept_: Contains the intercept (beta_0).

learned_coefficient = model.coef_[0]
learned_intercept = model.intercept_

print("--- Model Inspection ---")
print(f"Learned Coefficient (Slope): {learned_coefficient:.2f}")
print(f"Learned Intercept: {learned_intercept:.2f}")
print("The model learned the equation: y = {:.2f}x + {:.2f}".format(learned_coefficient, learned_intercept))

# 5. Make predictions
# The .predict() method uses the learned model to make predictions on new, unseen data.
# The input must be a 2D array, similar to the training data.
new_X = np.array([[6], [7]])
predictions = model.predict(new_X)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"Prediction for X = {val[0]}: {predictions[i]:.2f}")

# The .score() method can be used to evaluate the model's performance (R-squared value).
# R-squared is a measure of how well the model explains the variance in the data.
r_squared = model.score(X, y)
print(f"\nModel R-squared on training data: {r_squared:.4f}")
