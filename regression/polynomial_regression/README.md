# Polynomial Regression

Polynomial Regression is a form of regression analysis in which the relationship between the independent variable `x` and the dependent variable `y` is modeled as an *n*-th degree polynomial in `x`. While it models a non-linear relationship, it is still considered a special case of multiple linear regression because the model is linear in the parameters.

## How it Works

The core idea is to create new features by raising the existing features to a power. For example, if we have a single feature `x`, we can create new features `x^2`, `x^3`, `x^4`, etc. 

The equation for a polynomial regression model is:

**y = b + w₁x + w₂x² + ... + wₙxⁿ**

*   **y** is the predicted value.
*   **x** is the input value.
*   **b** is the bias (intercept).
*   **w₁, w₂, ..., wₙ** are the weights for each polynomial feature.

Even though the relationship between `x` and `y` is curved, the equation is linear with respect to the weights (`w`). This means we can use the same linear regression techniques (like the Normal Equation or Gradient Descent) to find the optimal weights for these new polynomial features.

### The Process

1.  **Feature Transformation:** The key step is to transform the original input features into polynomial features. For a degree of 2, a feature `x` becomes `[1, x, x^2]`.
2.  **Fit a Linear Model:** A standard linear regression model is then trained on this expanded set of features.

## Use Cases

Polynomial regression is useful when the data has a clear non-linear trend that can be captured by a polynomial function. It's often used in:

*   **Physics:** Modeling the trajectory of a projectile.
*   **Biology:** Analyzing the growth rate of bacteria.
*   **Economics:** Studying the relationship between variables that have a curved relationship.

This directory contains examples in Python and C++ demonstrating how to implement polynomial regression.
