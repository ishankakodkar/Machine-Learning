# Logistic Regression for Credit Default Prediction

Logistic Regression is a fundamental classification algorithm used to predict a binary outcome (e.g., yes/no, 1/0, true/false). Despite its name, it is used for classification, not regression. In finance, it's a workhorse model for tasks like predicting whether a borrower will default on a loan.

## Mathematical Formulation

Logistic Regression models the probability that an input \( X \) belongs to a particular category. It does this by passing the output of a linear equation through the **Sigmoid function** (or logistic function).

### The Sigmoid Function

The Sigmoid function squashes any real-valued number into a range between 0 and 1, which is perfect for representing a probability. The function is defined as:

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

Where \( z \) is the output of the linear model, \( z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \). The output, \( \sigma(z) \), is the predicted probability, \( P(y=1 | X) \).

### Cost Function: Log-Loss (Binary Cross-Entropy)

Unlike linear regression, we cannot use Mean Squared Error (MSE) as the cost function because the Sigmoid function would make it non-convex, leading to problems with optimization. Instead, we use the **Log-Loss** (or Binary Cross-Entropy) cost function, which is defined as:

\[ J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)] \]

Where:
-   \( m \) is the number of training examples.
-   \( y_i \) is the actual label of the i-th example (0 or 1).
-   \( \hat{p}_i \) is the predicted probability for the i-th example.

This cost function penalizes the model heavily for being confident and wrong. It is convex, ensuring that Gradient Descent can find the optimal set of coefficients (\( \beta \)).

## Application in Finance

-   **Credit Default Prediction:** Predict whether a loan applicant will default or not.
-   **Market Direction:** Classify whether a stock will go up or down.
-   **Fraud Detection:** Identify fraudulent transactions.
