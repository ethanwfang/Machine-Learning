## Clustering
Clustering is a form of unsupervised classification. The goal of clustering is to find a "natural grouping" in data. Examples in the same cluster should be more similar than those from other clusters. It can be thought of as "unsupervised classification." 

Clustering is typically used for exploratory data analysis. You don't want to split the data into training and test, and want to try and interpret the meaning of the clusters.

Input: nxm feature matrix X
Output: set of cluster assignments   
- Vector Representation: length $n$ vector $z$ where $z^{i} = j$    

Matrix representation: $n*k$ binary matrix W where $w^{i, j} = 1$
### K-Means Clustering
K-means clustering minimizes within-cluster SSE or cluster inertia.    

$ SSE = \sum_{i=1}^{n} \sum_{j=1}^{k} w^{(i, j)} ||x^{i} - \mu^{j}||^{2}_{2}$

So how can we find the optimal cluster assignments $W$?   
- $W$ is a binary matrix so this is a combinatorial optimization    
- Can't use gradient based methods!  

Introducing... Lloyd's algorithm or simply k-means algorithm:  
1. Randomly pick $k$ centroids from the examples as initial cluster centers
2. Assign each example to the nearest centroid, $\mu^{(i)}, j \in {1, ... , k}$   
3. Move the centroids to the center of the examples that were assigned to it  
4. Repeat steps 2 and 3 until the clsuter assignments do not change or a user-defined tolerance or maximum number of iterations is reached  

#### Model selection: choosing the number of clusters
K-means requires a selection of "num of clusters" as a hyperparameter, which is often unkown in practice. How can we choose k?  

A silhouette plot is a graphical measure of how tightly grouped the examples are in each cluster. A Silhouette coefficient $s^{(i)} \in [-1, 1]$ for a single example $x^{(i)}$:
1. Calculate the cluster cohesion, $a^{(i)}$, as the average distance between an example $x^{(i)}$ and all other points in the same cluster  
2. Calculate the cluster separation, $b^{(i)}$, from the next closest cluster as the average distance between the example $x^{(i)}$ and all examples in the nearest cluster.  
3. Calculate the silhouette, $s^{(i)}$ as the difference between cluster cohesion and separation divided by the greater of the two, as shown here:  
$s^{(i)} = \frac{b^{(i)} - a^{(i)}}{max[b^{(i)}, a^{(i)}]}$  

You want the silhouettes for each cluster to be around equal.

#### Internal vs. External clustering validation
- Silhouette coefficients are a form of internal validation. 


#### Assumptions of K-means
1. Assumes that everything is a spherical blob  
2. Attempts to minimize sum of squared Euclidean distances  
3. Scaling features so that the clusters are close to spherical helps to find better clusters   
4. K-means algorithm converges only to a local minimum of SSE  
5. 

### Hierarchical Clustering
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


