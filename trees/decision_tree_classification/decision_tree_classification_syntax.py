# This file demonstrates the syntax for using the DecisionTreeClassifier from scikit-learn.
# It focuses on the class, its key parameters for controlling tree growth, and how to visualize the tree.

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Create sample data
# Features: (technical indicator 1, technical indicator 2), Target: (0: Sell, 1: Buy)
X = np.array([[50, 20], [80, 90], [75, 15], [40, 30], [90, 85], [30, 10]])
y = np.array([0, 1, 1, 0, 1, 0])
feature_names = ['RSI', 'MACD']
class_names = ['Sell', 'Buy']

# 2. Instantiate the DecisionTreeClassifier model
# The DecisionTreeClassifier has several parameters to prevent overfitting:
# - criterion (str, default='gini'): The function to measure the quality of a split.
#   'gini' for Gini Impurity and 'entropy' for Information Gain.
# - max_depth (int, default=None): The maximum depth of the tree.
#   If None, then nodes are expanded until all leaves are pure or until they contain
#   less than min_samples_split samples. Used to control overfitting.
# - min_samples_split (int or float, default=2): The minimum number of samples required to split an internal node.
# - min_samples_leaf (int or float, default=1): The minimum number of samples required to be at a leaf node.

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3, # A shallow tree to keep it simple and prevent overfitting
    min_samples_split=2,
    random_state=42 # for reproducibility
)

# 3. Train the model
model.fit(X, y)

# 4. Inspect the model
# - .feature_importances_: The Gini importance of each feature.

print("--- Model Inspection ---")
for i, importance in enumerate(model.feature_importances_):
    print(f"Feature '{feature_names[i]}' Importance: {importance:.4f}")

# 5. Make predictions
new_X = np.array([[85, 88], [45, 25]])
predictions = model.predict(new_X)

print("\n--- Making Predictions ---")
for i, val in enumerate(new_X):
    print(f"For features {val}, Predicted Class: {class_names[predictions[i]]}")

# 6. Visualize the tree
# The plot_tree function from scikit-learn provides a clear visualization.
print("\n--- Visualizing Tree ---")
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
plt.title("Decision Tree Visualization")
# plt.show() # Uncomment to display the plot
print("A plot of the decision tree has been generated. If running in an interactive environment, uncomment plt.show() to display it.")

# To save the plot to a file:
plt.savefig('decision_tree_visualization.png')
print("Tree visualization saved to 'decision_tree_visualization.png'")
