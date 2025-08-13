# This file demonstrates the syntax for using Principal Component Analysis (PCA) from scikit-learn.
# It focuses on the class, its parameters, and how to interpret the results.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Create sample data
# Let's create data with 3 features, where two are highly correlated.
# This is a typical use case for PCA.
X = np.array([
    [1, 1, 5], 
    [2, 2.1, 6], 
    [3, 3.2, 7], 
    [4, 4.1, 8], 
    [5, 4.9, 9]
])

# 2. Standardize the data
# PCA is affected by scale, so you should scale the features before applying PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Instantiate the PCA model
# Key parameters for PCA:
# - n_components (int, float, or str): The number of components to keep.
#   - If int, it's the absolute number of components (e.g., 2).
#   - If float between 0.0 and 1.0, it's the amount of variance that needs to be explained
#     by the selected components (e.g., 0.95 for 95% variance).
#   - If 'mle', Minka's MLE is used to guess the dimension.
# - svd_solver (str, default='auto'): The solver to use.
#   'auto', 'full', 'arpack', 'randomized'. 'randomized' is often good for large datasets.

pca = PCA(n_components=2) # We want to reduce from 3 to 2 dimensions

# 4. Fit the model and transform the data
# .fit_transform() fits the model and applies the dimensionality reduction to the data.
X_pca = pca.fit_transform(X_scaled)

# 5. Inspect the results
# - .explained_variance_ratio_: Percentage of variance explained by each of the selected components.
# - .components_: The principal axes in feature space, representing the directions of maximum variance.

print("--- PCA Inspection ---")
print(f"Explained Variance Ratio per component: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained by {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.4f}")

print("\nPrincipal Components (Eigenvectors):")
print(pca.components_)

print("\n--- Transformed Data ---")
print("Original shape:", X_scaled.shape)
print("Transformed shape:", X_pca.shape)
print("\nTransformed data (first 5 rows):")
print(X_pca[:5])

# To see how much variance is explained by different numbers of components:
pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
# plt.show() # Uncomment to display
print("\nA plot showing cumulative explained variance has been generated. Uncomment plt.show() to display it.")
