# Decision Tree for Interpretable Trading Rules

Decision Trees are a non-parametric supervised learning method used for both classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. For finance, their main advantage is interpretability—they create a set of human-readable rules (e.g., "IF RSI > 70 AND MACD < 0, THEN Sell").

## Mathematical Formulation

Decision trees work by recursively partitioning the data into subsets based on the values of the input features. At each node of the tree, the algorithm chooses the feature and split point that results in the "best" separation of the classes.

### Splitting Criteria: Gini Impurity

To find the best split, the algorithm needs a way to measure the quality of a split. A common metric is **Gini Impurity**, which measures the frequency at which any element from the dataset will be mislabeled if it was randomly labeled according to the distribution of labels in the subset.

The Gini Impurity for a set of items with \( C \) classes is calculated as:

\[ G = 1 - \sum_{i=1}^{C} (p_i)^2 \]

Where \( p_i \) is the probability of an item being chosen from class \( i \). A Gini score of 0 represents a pure node (all elements belong to a single class), while a score of 0.5 (for a binary case) represents a completely impure node.

### Information Gain

The algorithm selects the split that results in the highest **Information Gain**, which is the difference between the impurity of the parent node and the weighted average impurity of the two child nodes.

\[ \text{Information Gain} = G_{\text{parent}} - \left( \frac{N_{\text{left}}}{N_{\text{parent}}} G_{\text{left}} + \frac{N_{\text{right}}}{N_{\text{parent}}} G_{\text{right}} \right) \]

This process is repeated at each node until a stopping criterion is met, such as the maximum depth of the tree is reached, the nodes become pure, or the number of samples in a node is too small to split further.

## Application in Finance

-   **Trading Signal Generation:** Create clear and explicit rules for entering or exiting trades based on technical indicators.
-   **Credit Scoring:** Build an interpretable model to accept or reject loan applications.
-   **Option Exercise Decisions:** Model the decision to exercise an American option before its expiration.
