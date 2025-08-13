# Linear Regression for Stock Price Prediction

Linear regression is a fundamental statistical and machine learning algorithm used for modeling the relationship between a dependent variable and one or more independent variables. In finance, it can be used as a simple baseline model for tasks like predicting stock prices based on historical data or other features.

## Mathematical Formulation

The goal of linear regression is to model the relationship between a dependent variable, \( y \), and one or more independent variables, \( X \).

For a simple linear regression with one independent variable, the model is represented by the equation:

\[ y = \beta_0 + \beta_1 x + \epsilon \]

Where:
-   \( y \) is the dependent variable (e.g., stock price).
-   \( x \) is the independent variable (e.g., a technical indicator or previous day's price).
-   \( \beta_0 \) is the y-intercept of the line (the value of \( y \) when \( x=0 \)).
-   \( \beta_1 \) is the slope of the line (the change in \( y \) for a one-unit change in \( x \)).
-   \( \epsilon \) is the error term, representing the part of \( y \) that is not explained by the model.

### Cost Function: Mean Squared Error (MSE)

To find the best-fit line, the model aims to minimize the **Mean Squared Error (MSE)**, which is the average of the squared differences between the actual values (\( y_i \)) and the predicted values (\( \hat{y}_i \)).

The formula for MSE is:

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

Where \( \hat{y}_i = \beta_0 + \beta_1 x_i \). The algorithm, typically using an optimization method like Gradient Descent, finds the values of \( \beta_0 \) and \( \beta_1 \) that minimize this cost function.

## Application in Finance

-   **Price Prediction:** Predict the next day's stock price based on the current day's open, high, low, or volume.
-   **Factor Modeling:** Determine the relationship between a stock's return and various market factors (e.g., the CAPM model).
-   **Arbitrage Trading:** Identify mispricings between two correlated assets.

While simple, linear regression provides a clear, interpretable model that is an excellent starting point for more complex analyses.
