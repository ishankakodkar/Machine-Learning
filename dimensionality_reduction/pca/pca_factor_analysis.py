import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load the dataset of stock returns
dataset = pd.read_csv('stock_returns.csv')
returns = dataset.iloc[:, 1:] # Drop the 'Date' column

# 2. Standardize the data
# It's important to scale the data before PCA
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# 3. Apply PCA
# We start by creating a PCA object that will keep all components
pca = PCA()
pca.fit(returns_scaled)

# 4. Analyze the results
explained_variance_ratio = pca.explained_variance_ratio_

print("PCA for Financial Factor Analysis")
print("=================================")
print(f"Explained variance by each component: {np.round(explained_variance_ratio, 3)}")

# The first component often represents the overall market movement
print(f"\nVariance explained by the first component (market factor): {explained_variance_ratio[0]*100:.2f}%")

# 5. Visualize the explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# 6. Transform the data to the principal components
# We can reduce the dimensionality by keeping only the first few components
pca_reduced = PCA(n_components=2)
principal_components = pca_reduced.fit_transform(returns_scaled)

pc_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

print("\nFirst 5 rows of the transformed data (Principal Components):")
print(pc_df.head())
