# Polynomial Regression in Quantitative Finance

Polynomial regression is an extension of linear regression that allows us to model non-linear relationships. In finance, this is particularly useful for capturing complex patterns that cannot be described by a straight line, such as the shape of a yield curve or the pricing of options.

## Mathematical Formulation

While it can model non-linear relationships, polynomial regression is still considered a form of linear regression because it is linear in its coefficients. The non-linearity is introduced by transforming the input features into polynomial features.

For a single independent variable \( x \), a polynomial regression model of degree \( d \) takes the form:

\[ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \epsilon \]

Where:
-   \( y \) is the dependent variable.
-   \( x, x^2, \dots, x^d \) are the features.
-   \( \beta_0, \beta_1, \dots, \beta_d \) are the coefficients learned by the model.
-   \( \epsilon \) is the error term.

This is achieved by first creating a new feature matrix consisting of \( [1, x, x^2, \dots, x^d] \) and then fitting a standard linear regression model to these new features. The same Mean Squared Error (MSE) cost function is used to find the optimal coefficients.

## Application in Finance

-   **Yield Curve Modeling:** The relationship between the yield of a bond and its maturity is often non-linear, making polynomial regression a suitable model.
-   **Options Pricing:** Model the non-linear relationship between an option's price and its underlying asset's price (the "smile" or "smirk" in implied volatility).
-   **Risk Analysis:** Capture non-linear relationships between a portfolio's return and various risk factors.
