# Logistic Regression for Credit Default Prediction

Logistic regression is a cornerstone of credit risk modeling in the financial industry. It is widely used to predict the probability that a borrower will default on a loan. This binary classification problem (default or no-default) is a perfect use case for logistic regression.

## Application in Finance

Financial institutions use logistic regression to:

-   **Assess Creditworthiness:** Determine the likelihood of a loan applicant defaulting based on their financial history, income, debt, and other factors.
-   **Set Interest Rates:** Price loans by assigning higher interest rates to applicants with a higher predicted probability of default.
-   **Manage Portfolio Risk:** Understand the overall risk profile of a loan portfolio by aggregating the default probabilities of individual loans.

## Model Formulation

The model calculates the probability of default, **P(Default=1)**, using the logistic (sigmoid) function. The input to this function is a linear combination of various borrower attributes (features).

$$ P(\text{Default}) = \frac{1}{1 + e^{-z}} $$

Where **z** is the weighted sum of the features:

$$ z = \beta_0 + \beta_1 \times (\text{Credit Score}) + \beta_2 \times (\text{Income}) + \beta_3 \times (\text{Loan Amount}) + \dots $$

The model learns the coefficients **β** that best separate the defaulters from the non-defaulters in the historical data.

### Cost Function

The cost function for logistic regression is different from that of linear regression because the hypothesis is non-linear (due to the sigmoid function). The cost function for a single training example is:

$$ Cost(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x)) $$

This is also known as the **Log Loss** or **Binary Cross-Entropy** loss. The goal of training is to find the parameters **θ** that minimize this cost function across all training examples.

### Gradient Descent

Similar to linear regression, we use gradient descent to find the optimal parameters. The update rule for the parameters is the same, but the hypothesis **h_θ(x)** is now the sigmoid function.
