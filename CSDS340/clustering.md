## Clustering

### Hierarhical Clustering
Hierarchical clustering algorithms group data points into clusters based on their similarity. Unlike other clustering methods like K-means, which requires a predefined number of clusters, hierarchical clustering buildsa hierarchy of clusteres and allows you to decide the number of clusteres at a later stage by cutting the dendrogram at a chosen level.

1. Agglomerative (Bottom-Up)
- Starts with each data point as its own cluster
- Iteratively merges the clsoest clusters based on a similarity or distance metric (e.g. Euclidean distance)
- Stops when all points are merged into a single cluster or whena specified number of clusters is reached.

2. Divisive (top-down)
- starts with all data points in a single cluster
- iteratively splits clusters into smaller clusters until each point is its own cluster or a stopping criterion is met

Steps in hierarchical clustering:
1. Compute Distance Matrix
2. Linkage Criteria
- Single linkage
- Average LInkage
- Maximum Linkage
- Ward's method
3. Merge or Split Clusters
4. Construct a Dendrogram

Advantages of hierarchical: does not require number of clusteres to be predefined, produces a hierarchy of clusters, which can give more insight into the data structure, can work well with small to medium-sized datasets

Disadvantages: computationally intensive for large datasets (b/c of distance matrix), sensitive to noise and outliers, which can distort the hierarchical structure, choice of distance metric and linkage method can significantly affect results

### Spectral Clustering
Spectral clustering is a graph-based clustering method that uses eigenvalues and eigenvectors of a graph's Lapalcian matrix to identify clusters. Unlike traditional clustering methods like K-means, spectral clustering excels at identifying non-convex clusters and clusters that may have irregular shapes.

### How does it work?
1. Represents data as a graph, where each data point is treated as a node in the graph
- edges represent the similarity between data points, which can be defined using Gaussian kernels or k-nearest neighbors.
- the result if a weighted adjacency matrix W, where w(i,j) represents the similarity between points i and j. 
2. Compute the Laplacian Matrix
- The Laplacian matrix captures the graph's structure. 
3. Solve Eigenvalue problem
- compute eigenvlues and eigenvectors of the Laplacian matrix
- select eigenvectors corresponding to the smallest k eigenvalues (excluding 0)
4. Embed the data
- use selected eigenvectors to embed the data points in a lower dimensional space (spectral space)
- each point is now represented as a vector in this new space
5. Apply Clustering
- Use traditional clustering algorithm on embedded data in the spectral space to group points into k clusters

Advantages: captures global structure, works on non-euclidean data, effective for irregular shapes
Disadvantages: Computationally expensive, not suitable for very large datasets, and requires choice of parameters (gamma for RBF, k for nearest neighbors)


