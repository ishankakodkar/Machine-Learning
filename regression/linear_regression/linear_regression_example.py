import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('google_stock_price.csv')

# For simplicity, we'll predict the next day's Close price based on the current day's Open price
X = dataset[['Open']]
y = dataset['Close'].shift(-1) # Shift the close price to get the next day's price

# Drop the last row since it has no corresponding next-day price
X = X[:-1]
y = y[:-1]

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the results
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color = 'red', label='Actual Price')
plt.plot(X_test, y_pred, color = 'blue', linewidth=2, label='Predicted Price')
plt.title('Google Stock Price Prediction (Linear Regression)')
plt.xlabel('Open Price')
plt.ylabel('Next Day Close Price')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# --- Part 1: Linear Regression from Scratch ---
print("--- Part 1: Linear Regression from Scratch ---")

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load data
data = pd.read_csv('salary_data.csv')
X_raw = data[['YearsExperience']].values
y_raw = data['Salary'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Fit the model
regressor_scratch = LinearRegressionScratch(learning_rate=0.02)
regressor_scratch.fit(X_train, y_train)

# Make predictions
y_pred_scratch = regressor_scratch.predict(X_test)

print(f"Scratch Model - Weights: {regressor_scratch.weights[0]:.2f}, Bias: {regressor_scratch.bias:.2f}")


# --- Part 2: Linear Regression with Scikit-learn ---
print("\n--- Part 2: Linear Regression with Scikit-learn ---")

# Create and train the model
regressor_sklearn = LinearRegression()
regressor_sklearn.fit(X_train, y_train)

# Make predictions
y_pred_sklearn = regressor_sklearn.predict(X_test)

print(f"Scikit-learn Model - Coeff: {regressor_sklearn.coef_[0]:.2f}, Intercept: {regressor_sklearn.intercept_:.2f}")

# Evaluate the scikit-learn model
mae = metrics.mean_absolute_error(y_test, y_pred_sklearn)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_sklearn))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# --- Part 3: Visualization ---

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', label='Actual Data')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Scikit-learn Prediction')
plt.plot(X_test, y_pred_scratch, color='blue', linestyle='--', linewidth=2, label='Scratch Prediction')

plt.title('Salary vs. Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.savefig('linear_regression_comparison.png')
print("\nGenerated plot: 'linear_regression_comparison.png'")
plt.show()
