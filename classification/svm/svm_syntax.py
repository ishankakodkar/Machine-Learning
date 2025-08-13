# This file demonstrates the syntax for using the Support Vector Classifier (SVC) from scikit-learn.
# It focuses on the class, its key parameters, and how to use it for non-linear classification.

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Create sample data
# Let's create non-linearly separable data (e.g., a 'moons' or 'circles' shape).
# This will highlight the strength of non-linear kernels like RBF.
X = np.array([
    [1, 2], [2, 3], [3, 3], [2, 1], [3, 2],  # Class 0
    [8, 9], [9, 10], [10, 10], [9, 8], [10, 9] # Class 1
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# It's highly recommended to scale data before using SVMs.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Instantiate the SVC model
# The SVC class has several critical parameters:
# - C (float, default=1.0): The regularization parameter. The strength of the regularization
#   is inversely proportional to C. It trades off correct classification of training
#   examples against maximization of the decision function's margin.
# - kernel (str, default='rbf'): Specifies the kernel type to be used in the algorithm.
#   'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'. 'rbf' is a good default.
# - gamma (str or float, default='scale'): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
#   'scale' (1 / (n_features * X.var())) is the default. 'auto' uses 1 / n_features.
# - degree (int, default=3): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

model = SVC(
    C=1.0,
    kernel='rbf', # Radial Basis Function is great for non-linear data
    gamma='scale',
    probability=True # Set to True to enable predict_proba
)

# 3. Train the model
# The .fit() method finds the optimal hyperplane.
model.fit(X_scaled, y)

# 4. Inspect the model
# - .support_vectors_: The data points that lie on the margin.
# - .n_support_: The number of support vectors for each class.

print("--- Model Inspection ---")
print(f"Number of support vectors for each class: {model.n_support_}")
# print(f"Support Vectors:\n{model.support_vectors_}") # Uncomment to see the vectors

# 5. Make predictions
# New data must be scaled with the same scaler.
new_X = np.array([[4, 4], [7, 7]])
new_X_scaled = scaler.transform(new_X)

class_predictions = model.predict(new_X_scaled)
probability_predictions = model.predict_proba(new_X_scaled)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"For features {val}:")
    print(f"  - Predicted Class: {class_predictions[i]}")
    print(f"  - Predicted Probabilities [P(0), P(1)]: {np.round(probability_predictions[i], 3)}")

# The .score() method returns the mean accuracy.
accuracy = model.score(X_scaled, y)
print(f"\nModel Accuracy on training data: {accuracy:.4f}")
