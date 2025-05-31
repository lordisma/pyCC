import numpy as np

from ..model import BaseEstimator
from ..utils.distance import match_distance

import logging
logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        constraints,
        n_clusters = 8,
        init = "random",
        distance = "euclidean",
        custom_initial_centroids = None,
        tol = 1e-4,
        max_iter=300,
        limit = 1,
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
        friends = np.argwhere(self.constraints[:, d] > 0)
        strangers = np.argwhere(self.constraints[:, d] < 0)
        in_cluster = np.argwhere(self._labels == c)

        friends = np.sum(np.isin(friends, in_cluster))
        strangers = np.sum(np.isin(strangers, in_cluster))

        return friends - strangers

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

        self._delta = self.calculte_delta(aux)

    def _update(self):
        """Update the centroids.

        This method update the centroids of the clusters, also calculate the amount of empty clusters and
        mark them to be removed.

        Returns
        _______
        to_remove: numpy.ndarray
            The centroids to remove.
        """
        to_remove = np.array([False] * self.n_clusters)
        for centroid in range(self.n_clusters):
            assigned = np.where(self._labels == centroid)

            if not np.any(assigned):
                to_remove[centroid] = True
                logger.debug(f"Centroid {centroid} is empty, marking for removal")
                continue

            if np.sum(assigned) < 2:
                self.centroids[centroid] = np.mean(self.X[assigned], axis=0)

                if np.any(np.isnan(self.centroids[centroid])):
                    raise ValueError(f"Centroid {centroid} has NaN values, crashing the algorithm")
                
            elif np.sum(assigned) == 1:
                logger.debug(f"Centroid {centroid} has only one instance, reinitializing")
                self.centroids[centroid] = np.random.normal(self.X[assigned][0], scale=0.1, size=self.X.shape[1])

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
                logger.debug(f"Removing {centroid} empty centroids")
                removed += 1
                continue

            self._labels[np.where(self._labels == centroid)] = centroid - removed
        
        self.centroids = self.centroids[~to_remove]
        self.n_clusters = self.centroids.shape[0]

    def get_penalties(self, idx: int, iteration: int) -> np.ndarray:
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
        instance = self.X[idx]

        diff = self.centroids - np.repeat(instance[np.newaxis, :], self.n_clusters, axis=0)
        distances = self.distance(diff, axis=1).flatten()
        diff_allies = np.array([self.diff_alliances(idx, c) for c in range(self.n_clusters)])

        xi = self.x0 * (self.rate**iteration)
        return distances - (xi * diff_allies)
              

    def _fit(self):
        logger.debug("Fitting RDPM model")

        iteration = 0
        while not self.stop_criteria(iteration):
            iteration += 1

            for d in np.arange(self.X.shape[0]):
                penalties = self.get_penalties(d, iteration)
                label = np.argmin(penalties)

                if penalties[label] >= self.limit:
                    logger.debug(f"Instance {d} exceeds limit, creating new cluster")
                    self.n_clusters += 1
                    self.centroids = np.vstack((self.centroids, self.X[d, :]))
                    label = self.centroids.shape[0] - 1

                self._labels[d] = label

            logger.debug(f"Iteration {iteration} completed with clusters: {self.n_clusters}")
            self.update()
