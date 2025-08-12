# Linear Regression for Stock Price Prediction

Linear regression is a fundamental algorithm in quantitative finance, often used as a baseline for predicting asset prices. In this example, we'll use it to model the relationship between a stock's past performance and its future price.

## Application in Finance

While simple linear regression has its limitations in the complex world of financial markets, it serves as an excellent starting point for understanding more advanced models. It can be used to:

-   Identify trends in stock prices.
-   Model the relationship between a stock and a market index (e.g., calculating Beta).
-   Predict future returns based on historical data or other factors.

## Mathematical Formulation

The core idea is to model the stock price (or its return) as a linear function of one or more predictor variables. For a simple model where we predict tomorrow's price based on today's price, the equation is:

$$ y = \beta_0 + \beta_1 x + \epsilon $$

Where:
- **y** is the predicted price for the next day.
- **x** is the price of the stock today.
- **β₀** and **β₁** are the coefficients determined by the model.
- **ε** is the error term.

More complex models could include other features like trading volume, moving averages, or macroeconomic indicators.


Linear regression is one of the most fundamental algorithms in supervised machine learning. It's used to predict a continuous target variable (like price, temperature, or salary) based on one or more predictor variables.

## How it Works

The goal of linear regression is to find the best-fitting straight line through the data points. This line is called the regression line.

The equation for a simple linear regression line (with one predictor variable) is:

**y = mx + b**

*   **y** is the predicted value (dependent variable).
*   **x** is the input value (independent variable).
*   **m** is the slope of the line (weight).
*   **b** is the y-intercept (bias).

The algorithm's job is to find the optimal values for **m** and **b** that minimize the difference between the predicted values and the actual values in the training data. This difference is measured by a **loss function**, most commonly the **Mean Squared Error (MSE)**.

### Optimization: Gradient Descent

To find the best `m` and `b`, we use an optimization algorithm called **Gradient Descent**. It works by:

1.  Initializing `m` and `b` with random values.
2.  Calculating the gradient (the partial derivatives) of the loss function with respect to `m` and `b`.
3.  Updating `m` and `b` by taking a small step in the direction of the negative gradient.
4.  Repeating steps 2 and 3 until the loss stops decreasing significantly (convergence).

## Use Cases

*   **Business:** Predicting sales, analyzing the impact of advertising.
*   **Finance:** Forecasting stock prices, assessing risk.
*   **Real Estate:** Estimating house prices based on features like size and location.

This directory contains a Python script demonstrating how to implement linear regression both from scratch and using the `scikit-learn` library.
