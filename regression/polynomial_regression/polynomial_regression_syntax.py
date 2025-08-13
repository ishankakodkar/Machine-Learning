# This file demonstrates the syntax for performing Polynomial Regression using scikit-learn.
# The key is to use the `PolynomialFeatures` transformer in combination with `LinearRegression`.

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Create sample data
# Let's create data that follows a quadratic relationship: y = 0.5*x^2 - 2*x + 5
X = np.array([[0], [1], [2], [3], [4]])
y = np.array([5, 3.5, 3, 3.5, 5])

# 2. Create Polynomial Features
# The `PolynomialFeatures` class generates a new feature matrix consisting of all
# polynomial combinations of the features with degree less than or equal to the specified degree.
# - degree (int, default=2): The degree of the polynomial features.
# - include_bias (bool, default=True): If True, include a bias column (feature of all ones).

poly_features = PolynomialFeatures(
    degree=2,         # We choose 2 because we know the underlying relationship is quadratic
    include_bias=False  # We set this to False because LinearRegression handles the intercept
)

# 3. Transform the original features
# The .fit_transform() method creates the new polynomial features.
# For X = [[x]], a degree-2 transformation will produce [[x, x^2]].
X_poly = poly_features.fit_transform(X)

print("--- Feature Transformation ---")
print("Original features (X):")
print(X)
print("\nTransformed features (X_poly) with degree=2:")
print(X_poly)

# 4. Fit a Linear Regression model on the new features
# We now treat this as a multiple linear regression problem where the features are x and x^2.
model = LinearRegression()
model.fit(X_poly, y)

# 5. Inspect the model
# The coefficients will correspond to the polynomial terms [x, x^2]
learned_coeffs = model.coef_
learned_intercept = model.intercept_

print("\n--- Model Inspection ---")
print(f"Learned Coefficients (for x, x^2): {learned_coeffs}")
print(f"Learned Intercept: {learned_intercept:.2f}")
print("The model learned the equation: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(learned_coeffs[1], learned_coeffs[0], learned_intercept))

# 6. Make predictions
# To make a prediction, the new data must also be transformed using the same
# `poly_features` transformer.
new_X = np.array([[5]])
new_X_poly = poly_features.transform(new_X)
prediction = model.predict(new_X_poly)

print("\n--- Making Predictions ---")
print(f"Prediction for X = 5: {prediction[0]:.2f}")
