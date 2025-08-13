# This file demonstrates the syntax for using the RandomForestClassifier from scikit-learn.
# It focuses on the class and its key parameters for building an ensemble of trees.

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Create sample data
# Features: (indicator 1, indicator 2, indicator 3), Target: (0: Sell, 1: Buy)
X = np.array([[50, 20, 1], [80, 90, 0], [75, 15, 1], [40, 30, 0], [90, 85, 1], [30, 10, 0]])
y = np.array([0, 1, 1, 0, 1, 0])
feature_names = ['RSI', 'MACD', 'Volume_Spike']
class_names = ['Sell', 'Buy']

# 2. Instantiate the RandomForestClassifier model
# Key parameters for Random Forest:
# - n_estimators (int, default=100): The number of trees in the forest.
#   More trees can increase performance but also increase computation time.
# - criterion (str, default='gini'): The function to measure the quality of a split.
#   'gini' or 'entropy'.
# - max_depth (int, default=None): The maximum depth of the trees.
# - max_features (int, float or {"auto", "sqrt", "log2"}, default='auto'):
#   The number of features to consider when looking for the best split.
#   'sqrt' is a common choice (sqrt(n_features)).
# - oob_score (bool, default=False): Whether to use out-of-bag samples to estimate
#   the generalization accuracy. This is a form of cross-validation using the data
#   left out of the bootstrap sample for each tree.
# - n_jobs (int, default=None): The number of jobs to run in parallel. -1 means using all processors.

model = RandomForestClassifier(
    n_estimators=100, # Using 100 trees in the forest
    max_features='sqrt', # Consider sqrt(n_features) at each split
    max_depth=5,
    oob_score=True, # Use out-of-bag score for validation
    random_state=42, # For reproducibility
    n_jobs=-1
)

# 3. Train the model
model.fit(X, y)

# 4. Inspect the model
# - .feature_importances_: The Gini importance of each feature, averaged over all trees.
# - .oob_score_: The score of the training dataset obtained using an out-of-bag estimate.

print("--- Model Inspection ---")
print(f"Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")
print("\nFeature Importances:")
for i, importance in enumerate(model.feature_importances_):
    print(f"  - Feature '{feature_names[i]}': {importance:.4f}")

# 5. Make predictions
new_X = np.array([[85, 88, 1], [45, 25, 0]])
predictions = model.predict(new_X)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"For features {val}, Predicted Class: {class_names[predictions[i]]}")
