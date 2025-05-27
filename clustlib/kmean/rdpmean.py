import numpy as np
from scipy import spatial as sdist
from typing import Tuple

from ..model import BaseEstimator
from ..utils.distance import match_distance


class RDPM(BaseEstimator):
    """RDPM

    Parameters
    __________
    n_clusters: int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init: {'random', 'custom'}, default='random'
        Method for initialization, defaults to 'random':
    distance: {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        The distance metric to use. See `scipy.spatial.distance` for a list of available metrics.
    max_iter: int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive
        iterations to declare convergence.
    custom_initial_centroids: numpy.ndarray, default=None
        Custom initial centroids to be used in the initialization. Only used if init='custom'.
    limit: float, default=0.005
        The limit of distance with the closest centroid allowed to not create another cluster
    x0: float, default=0.001
        The initial value of xi, which would prevent the creation of new clusters
    rate: float, default=2.0
        The rate of increase of xi, which would prevent the creation of new clusters
    """

    x0: float
    rate: float
    limit: float


    def initialize(
        self,
        constraints,
        n_clusters = 8,
        init = "random",
        distance = "euclidean",
        custom_initial_centroids = None,
        tol = 1e-4,
        max_iter=300,
        limit = 0.005,
        x0 = 0.001,
        rate = 2.0
    ):
        self.n_clusters = n_clusters
        self.constraints = constraints
        self.init = init
        self.distance = match_distance(distance)
        self.centroids = None
        self.custom_initial_centroids = None
        self.max_iter = max_iter
        self.tol = 0.0001
        self.limit = limit
        self.x0 = x0
        self.rate = rate
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids

    def diff_alliances(self, d, c) -> int:
        """Calculate the difference of alliances.

        This method calculates how many friends and strangeres are in the centroid given as parameter.

        Parameters
        __________
        d: int
            The index of the instance to be predicted

        c: int
            The index of the centroid to be predicted

        Returns
        _______
        diff: int
            The difference of alliances.
        """
        friends = np.where(self.constraints[:, d] == 1)
        strangers = np.where(self.constraints[:, d] == -1)

        in_cluster = np.where(self._labels[friends] == c)

        friends = np.logical_and(friends, in_cluster)
        strangers = np.logical_and(strangers, in_cluster)

        return np.sum(friends) - np.sum(strangers)

    def update(self):
        """Override the update method.

        This method overrides the update method of the BaseEstimator, in order to update the 
        delta of the centroids and consider the empty clusters.
        """
        aux = np.copy(self.centroids)
        to_remove = self._update()

        if np.any(to_remove):
            self.__delete_centroids(to_remove)
            np.delete(aux, to_remove, axis=0)

        self.calculate_delta(aux)

    def _update(self):
        """Update the centroids.

        This method update the centroids of the clusters, also calculate the amount of empty clusters and
        mark them to be removed.

        Returns
        _______
        to_remove: numpy.ndarray
            The centroids to remove.
        """
        to_remove = np.arange([False] * self.n_clusters)
        for centroid in range(self.n_clusters):
            assigned = np.where(self._labels == centroid)

            if np.any(assigned):
                to_remove[centroid] = True
                continue

            self.centroids[centroid] = np.mean(
                self.X[assigned], axis=0
            )

        return to_remove

    def __delete_centroids(self, to_remove):
        """Delete centroids that are empty.

        Parameters
        __________
        to_remove: numpy.ndarray
            The centroids to remove.
        """
        if np.any(to_remove):
            return
        
        removed = 0
        for centroid, is_empty in enumerate(to_remove):
            if is_empty:
                removed += 1
                continue

            self._labels[np.where(self._labels == centroid)] = centroid - removed
            
        self.centroids = np.delete(self.centroids, to_remove, axis=0)
        self.n_clusters = np.sum(to_remove)

    def get_penalties(self, instance: int, iteration: int) -> np.ndarray:
        """Get the label at which the instance should be assigned.

        Parameters
        __________
        instance: int
            The index of the instance to be predicted

        iteration: int
            The iteration of the algorithm

        Returns
        _______
        penalties: numpy.ndarray
            The penalties of the instance to be assigned to each centroid.
        """
        diff_allies = np.array([self.diff_alliances(instance, c) for c in range(self.n_clusters)])
        distances = np.array([self.distance(instance, c) for c in range(self.n_clusters)])
        xi = self.x0 * (self.rate**iteration)

        penalty = distances - (xi * diff_allies)
        return np.argmin(penalty)
              

    def fit(self, dataset: np.ndarray, labels: np.array = None):
        self.X = dataset
        
        if labels:
            self._labels = labels
        else: 
            self._labels = np.random.randint(0, self.n_clusters, self.X.shape[0])

        iteration = 0
        while not self.stop_criteria(iteration):
            iteration += 1

            for d in np.arange(self.X.shape[0]):
                penalties = self.get_penalties(d, iteration)
                label = np.argmin(penalties)

                if penalties[label] >= self.limit:
                    self.n_clusters += 1
                    self.centroids = np.vstack((self.centroids, self.X[d, :]))
                    label = self.centroids.shape[0] - 1

                self._labels[d] = label
            self.update()
