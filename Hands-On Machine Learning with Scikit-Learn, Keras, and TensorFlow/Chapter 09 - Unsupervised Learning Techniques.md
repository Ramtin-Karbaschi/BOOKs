# Chapter 9: Unsupervised Learning Techniques

## **Foundations of Unsupervised Learning**

Unsupervised learning uncovers hidden patterns in unlabeled data by discovering structures, relationships, or clusters. It is particularly useful when labeled data is scarce or unavailable.

### **Key Motivations**

1. **Clustering**:
    - Identify natural groupings within the data.
2. **Dimensionality Reduction**:
    - Simplify high-dimensional data while preserving its structure.
3. **Anomaly Detection**:
    - Identify outliers or rare events that deviate from normal patterns.

---

## **Clustering Algorithms: Grouping Data Points**

Clustering partitions data into groups based on similarity, enabling tasks like customer segmentation, image compression, and anomaly detection.

### **K-Means Clustering**

- **Purpose**: Partition data into $k$ clusters by minimizing the variance within each cluster.
- **Objective Function**:
    
    $$
    J = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2
    $$
    
- ***Implementation**:*
    
    ```python
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    ```
    
- **Strengths**:
    - Simple and scalable.
    - Effective for spherical clusters.
- **Limitations**:
    - Struggles with irregularly shaped clusters.
    - Requires specifying $k$ in advance.

### **Hierarchical Clustering**

- **Purpose**: Build a hierarchy of clusters using agglomerative (bottom-up) or divisive (top-down) approaches.
- **Dendrogram Visualization**:
    
    ```python
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    Z = linkage(X, method='ward')
    dendrogram(Z)
    plt.show()
    ```
    
- **Strengths**:
    - Does not require specifying $k$ beforehand.
    - Provides a visual representation of cluster relationships.
- **Limitations**:
    - Computationally expensive for large datasets.

### **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

- **Purpose**: Group points based on density, identifying clusters of arbitrary shapes and marking outliers as noise.
- ***Implementation**:*
    
    ```python
    from sklearn.cluster import DBSCAN
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    ```
    
- **Strengths**:
    - Robust to noise.
    - Capable of handling irregularly shaped clusters.
- **Limitations**:
    - Sensitive to hyperparameters like `eps` and `min_samples`.

---

## **Dimensionality Reduction for Clustering**

High-dimensional data often suffers from sparsity and noise, making clustering challenging. Dimensionality reduction techniques like PCA or t-SNE simplify the data while preserving its structure.

### **PCA for Clustering**

- Reduce dimensions to improve clustering performance:
    
    ```python
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_reduced)
    ```
    

### **t-SNE for Visualizing Clusters**

- Reduce dimensions to 2D or 3D for visualization:
    
    ```python
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
    plt.show()
    ```
    

---

## **Anomaly Detection: Identifying Rare Events**

Anomaly detection identifies outliers or rare events that deviate significantly from normal patterns.

### **Statistical Methods**

- **Z-Score**:
    
    $$
    z = \frac{x - \mu}{\sigma}
    $$
    
    - Outliers are identified as points with $|z| > 3$.
- **Gaussian Distribution**:
Identify anomalies based on deviations from expected behavior:
    
    ```python
    from scipy.stats import norm
    
    mean, std = np.mean(X), np.std(X)
    threshold = norm.ppf(0.999, loc=mean, scale=std)
    anomalies = X[X > threshold]
    ```
    

### **Machine Learning Approaches**

- **Isolation Forest**:
Detect anomalies in high-dimensional data:
    
    ```python
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(contamination=0.01)
    anomalies = iso_forest.fit_predict(X)
    ```
    
- **One-Class SVM**:
Identify outliers using a decision boundary:
    
    ```python
    from sklearn.svm import OneClassSVM
    
    oc_svm = OneClassSVM(nu=0.01)
    anomalies = oc_svm.fit_predict(X)
    ```
    

---

## **Practical Implementation**

This section provides hands-on examples for implementing unsupervised learning techniques.

**Example:**

*K-Means for Customer Segmentation*

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(customer_data)

plt.scatter(customer_data[:, 0], customer_data[:, 1], c=labels)
plt.title("Customer Segmentation")
plt.show()
```

*DBSCAN for Spatial Data*

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(spatial_data)

plt.scatter(spatial_data[:, 0], spatial_data[:, 1], c=labels)
plt.title("DBSCAN Clustering")
plt.show()
```

*Anomaly Detection for Fraud*

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.01)
anomalies = iso_forest.fit_predict(transaction_data)

frauds = transaction_data[anomalies == -1]
print("Detected Frauds:", frauds)
```

---

## **Applications and Use Cases**

Unsupervised learning has diverse applications:

- **Healthcare**: Identify patient subgroups for personalized treatment.
- **Finance**: Detect fraudulent transactions or unusual market behavior.
- **E-commerce**: Segment customers for targeted marketing campaigns.

---

## **Conclusion**

This chapter equips readers with tools to uncover hidden patterns in unlabeled data. By mastering clustering, dimensionality reduction, and anomaly detection, you'll be prepared to analyze complex datasets effectively.

### **Key Takeaways**

- K-Means is effective for spherical clusters but requires specifying $k$.
- Hierarchical clustering provides a dendrogram for visualizing relationships.
- DBSCAN handles irregularly shaped clusters and identifies noise.
- Dimensionality reduction enhances clustering performance and interpretability.
- Anomaly detection identifies outliers using statistical and machine learning methods.