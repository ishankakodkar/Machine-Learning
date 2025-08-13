# This file demonstrates the syntax for using the LogisticRegression model from scikit-learn.
# It focuses on the class, its parameters, and how to use it for binary classification.

import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Create sample data
# Let's create a simple dataset for binary classification.
# Features: (age, income), Target: (0: no purchase, 1: purchase)
X = np.array([[25, 40000], [35, 60000], [45, 80000], [20, 20000], [40, 120000], [50, 30000]])
y = np.array([0, 0, 1, 0, 1, 1]) # 0 = No, 1 = Yes

# 2. Instantiate the LogisticRegression model
# The LogisticRegression class has several important parameters:
# - penalty (str, default='l2'): Specifies the norm used in the penalization (regularization).
#   'l1', 'l2', 'elasticnet', or 'none'.
# - C (float, default=1.0): Inverse of regularization strength; must be a positive float.
#   Smaller values specify stronger regularization.
# - solver (str, default='lbfgs'): Algorithm to use in the optimization problem.
#   'liblinear' is good for small datasets. 'sag' and 'saga' are faster for large ones.
#   'newton-cg', 'lbfgs' handle multinomial loss. 'saga' also supports elasticnet penalty.
# - max_iter (int, default=100): Maximum number of iterations for solvers to converge.

model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear' # A good choice for small datasets
)

# 3. Train the model
# The .fit() method trains the model on the provided data.
model.fit(X, y)

# 4. Inspect the learned parameters
# - .coef_: Contains the coefficients for each feature.
# - .intercept_: Contains the intercept.

print("--- Model Inspection ---")
print(f"Coefficients (for age, income): {model.coef_}")
print(f"Intercept: {model.intercept_}")

# 5. Make predictions
# The .predict() method predicts the class label (0 or 1).
# The .predict_proba() method predicts the probability for each class [P(0), P(1)].
new_X = np.array([[30, 50000], [55, 75000]])

class_predictions = model.predict(new_X)
probability_predictions = model.predict_proba(new_X)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"For features {val}:")
    print(f"  - Predicted Class: {class_predictions[i]}")
    print(f"  - Predicted Probabilities [P(0), P(1)]: {np.round(probability_predictions[i], 3)}")

# The .score() method returns the mean accuracy on the given test data and labels.
accuracy = model.score(X, y)
print(f"\nModel Accuracy on training data: {accuracy:.4f}")
