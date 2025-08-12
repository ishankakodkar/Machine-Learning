import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
dataset = pd.read_csv('yield_curve_data.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Create polynomial features
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Fit the Polynomial Regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.figure(figsize=(10,6))
plt.scatter(X, y, color = 'red', label='Actual Yield')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue', label='Fitted Yield Curve')
plt.title('Yield Curve Modeling (Polynomial Regression)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield')
plt.legend()
plt.grid(True)
plt.show()

# Predicting a new result with Polynomial Regression
yield_prediction = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Predicted yield for a 6.5-year maturity: {yield_prediction[0]:.4f}")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- Part 1: Load and Prepare Data ---
print("--- Loading Data ---")
data = pd.read_csv('position_salaries.csv')
X = data[['Level']].values
y = data['Salary'].values

print("First 5 rows of data:")
print(data.head())

# --- Part 2: Simple Linear Regression (for comparison) ---
print("\n--- Training Simple Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# --- Part 3: Polynomial Regression ---
print("--- Training Polynomial Regression ---")

# Create polynomial features (e.g., x, x^2, x^3, ...)
# The degree determines how complex the curve can be.
degree = 4
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Train a linear regression model on the polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# --- Part 4: Visualization ---
print("\n--- Visualizing Results ---")

# Create a smoother curve for plotting the polynomial model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.figure(figsize=(10, 6))
# Scatter plot of the actual data
plt.scatter(X, y, color='red', label='Actual Data')

# Plot the simple linear regression line
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Fit')

# Plot the polynomial regression curve
plt.plot(X_grid, poly_reg.predict(poly_features.transform(X_grid)), color='green', label=f'Polynomial Fit (Degree {degree})')

plt.title('Salary vs. Position Level (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.savefig('polynomial_regression.png')
print("Generated plot: 'polynomial_regression.png'")
plt.show()

# --- Part 5: Prediction ---
print("\n--- Making a Prediction ---")
# Predict the salary for a level 6.5
level_to_predict = 6.5
level_poly = poly_features.transform([[level_to_predict]])
predicted_salary = poly_reg.predict(level_poly)

print(f"Predicted salary for level {level_to_predict}: ${predicted_salary[0]:,.2f}")
