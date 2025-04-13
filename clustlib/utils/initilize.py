import numpy as np

def random(dataset: np.ndarray, n_clusters: int) -> np.ndarray:
    """Get random centroids from the dataset.

    Parameters
    __________
    dataset: numpy.ndarray
        The data to cluster.
    n_clusters: int
        The number of clusters.

    Returns
    _______
    centroids: numpy.ndarray
        The centroids of the clusters.
    """
    if n_clusters > dataset.shape[0]:
        raise ValueError("Number of clusters cannot be greater than number of samples.")
    
    max = np.max(dataset, axis=0)
    min = np.min(dataset, axis=0)

    if np.array_equal(max, min):
        raise ValueError("Dataset has no variance. Cannot initialize centroids.")
    
    return np.random.uniform(min, max, (n_clusters, dataset.shape[1]))

def kmeans(dataset: np.ndarray, n_clusters: int) -> np.ndarray:
    """Get the centroids of the clusters using k-means.

    Parameters
    __________
    dataset: numpy.ndarray
        The data to cluster.
    n_clusters: int
        The number of clusters.

    Returns
    _______
    centroids: numpy.ndarray
        The centroids of the clusters.
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100).fit(dataset)
    return kmeans.cluster_centers_