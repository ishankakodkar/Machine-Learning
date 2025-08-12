# K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to find groups in unlabeled data. The "K" in K-Means represents the number of clusters. The algorithm works by iteratively assigning each data point to one of K groups based on the features provided.

## How it Works

The K-Means algorithm follows these steps:

1.  **Initialization**: Randomly select K data points from the dataset to serve as the initial centroids (cluster centers).
2.  **Assignment Step**: Assign each data point to the nearest centroid. The distance is typically measured using the Euclidean distance.
3.  **Update Step**: Recalculate the centroids by taking the mean of all data points assigned to each cluster.
4.  **Repeat**: Repeat the assignment and update steps until the centroids no longer change significantly, or a maximum number of iterations is reached.

### The Objective Function

K-Means aims to minimize the within-cluster sum of squares (WCSS), also known as inertia. The WCSS is the sum of the squared distances between each data point and its assigned centroid.

$$ WCSS = \sum_{j=1}^{K} \sum_{i=1}^{n_j} ||x_i^{(j)} - c_j||^2 $$

Where:
- **K** is the number of clusters.
- **n_j** is the number of data points in cluster j.
- **x_i⁽ʲ⁾** is the i-th data point in cluster j.
- **c_j** is the centroid of cluster j.

### Choosing the Right K

A common method for choosing the optimal value for K is the **Elbow Method**. This involves running the K-Means algorithm for a range of K values and plotting the WCSS for each. The "elbow" of the curve—the point where the rate of decrease in WCSS sharply changes—is often considered the optimal K.
