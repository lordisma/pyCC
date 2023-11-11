"""KMeans estimator
This file contains the KMeans estimator implementation.

Notes
_____
The KMeans estimator is a clustering algorithm that aims to partition n observations into k clusters in which each
observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a
partitioning of the data space into Voronoi cells.

Based on the Elkan's algorithm, using triangle inequality, the KMeans estimator is able to reduce the number of
distance calculations between points and centroids, improving the performance of the algorithm.

Contains
________
    KMeans: KMeans estimator implementation.
"""

import numpy as np
from .. import BaseEstimator


class KMeans(BaseEstimator):
    """KMeans estimator
    KMeans estimator implementation.

    Parameters
    __________
    n_clusters: int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init: {'random', 'custom'}, default='random'
        Method for initialization, defaults to 'random':
            'random': choose k observations (rows) at random from data for the initial centroids.
            'custom': use custom_initial_centroids as initial centroids.
    max_iter: int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive
        iterations to declare convergence.
    custom_initial_centroids: numpy.ndarray, default=None
        Custom initial centroids to be used in the initialization. Only used if init='custom'.
    """

    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None

    def fit(self, dataset):
        """Compute k-means clustering.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """
        # Initialize centroids
        if self.init == "random":
            self.centroids = self._init_random(dataset)
        elif self.init == "custom":
            self.centroids = self._init_custom()
        else:
            raise ValueError(f"Invalid init method: '{self.init}'")

        # Initialize clusters
        return self.centroids

    def _init_random(self, dataset):
        """Initialize centroids randomly.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.

        Returns
        _______
        numpy.ndarray
            Set of centroids
        """
        # Randomly choose k observations (rows) at random from data for the initial centroids.
        random_indices = np.random.choice(
            dataset.shape[0], self.n_clusters, replace=False
        )
        return dataset[random_indices, :]

    def _init_custom(self):
        """Initialize centroids using custom_initial_centroids.

        Returns
        _______
        numpy.ndarray
            Set of centroids
        """
        return self.custom_initial_centroids
