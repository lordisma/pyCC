import logging

import numpy as np

from ..model import BaseEstimator
from ..utils.distance import match_distance

logger = logging.getLogger(__name__)


class RDPM(BaseEstimator):
    """RDPM clustering algorithm.

    Args:
        constraints (numpy.ndarray): The constraints matrix.
        n_clusters (int, optional): The number of clusters to form. Defaults to 8.
        init (str, optional): Initialization method ('random' or 'custom'). 
            Defaults to 'random'.
        distance (str, optional): Distance metric ('euclidean', 'manhattan', 'cosine'). 
            Defaults to 'euclidean'.
        custom_initial_centroids (numpy.ndarray, optional): Custom initial centroids. 
            Used if init='custom'. Defaults to None.
        tol (float, optional): Convergence tolerance. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        limit (float, optional): Distance limit for creating new clusters. 
            Defaults to 1.
        x0 (float, optional): Initial value of xi for preventing new clusters. 
            Defaults to 0.001.
        rate (float, optional): Rate of increase of xi. Defaults to 2.0.

    """

    x0: float
    rate: float
    limit: float

    def __init__(
        self,
        constraints,
        n_clusters=8,
        init="random",
        distance="euclidean",
        custom_initial_centroids=None,
        tol=1e-4,
        max_iter=300,
        limit=1,
        x0=0.001,
        rate=2.0,
    ):
        self.n_clusters = n_clusters
        self.constraints = constraints
        self.init = init
        self.distance = match_distance(distance)
        self.centroids = None
        self.max_iter = max_iter
        self.tol = tol
        self.limit = limit
        self.x0 = x0
        self.rate = rate
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids

    def diff_alliances(self, d, c) -> int:
        """Calculate the difference of alliances.

        Calculates the "friends" and "strangers" for the instance `d` in the 
        cluster `c`. Friends are instances that are positively constrained with `d`, 
        and strangers are instances that are negatively constrained with `d`. The 
        difference is calculated as the number of friends minus the number of strangers
        in the cluster.

        Args:
            d (int): Index of the instance to be predicted.
            c (int): Index of the centroid to be predicted.

        Returns:
            int: Difference of alliances.

        """
        friends = np.argwhere(self.constraints[:, d] > 0)
        strangers = np.argwhere(self.constraints[:, d] < 0)
        in_cluster = np.argwhere(self._labels == c)

        friends = np.sum(np.isin(friends, in_cluster))
        strangers = np.sum(np.isin(strangers, in_cluster))

        return friends - strangers

    def update(self):
        """Override the update method.

        Updates the centroids and considers empty clusters.
        """
        aux = np.copy(self.centroids)
        to_remove = self._update()

        if np.any(to_remove):
            self._delete_centroids(to_remove)
            aux = aux[~to_remove]

        self._delta = self.calculte_delta(aux)

    def _update(self):
        """Update the centroids.

        Updates the centroids of the clusters and marks empty clusters for removal.

        Returns:
            numpy.ndarray: Array indicating centroids to remove.

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
                    raise ValueError(
                        f"Centroid {centroid} has NaN values, crashing the algorithm"
                    )

            elif np.sum(assigned) == 1:
                logger.debug(
                    f"Centroid {centroid} has only one instance, reinitializing"
                )
                self.centroids[centroid] = np.random.normal(
                    self.X[assigned][0], scale=0.1, size=self.X.shape[1]
                )

        return to_remove

    def _delete_centroids(self, to_remove):
        """Delete centroids that are empty.

        Args:
            to_remove (numpy.ndarray): Array indicating centroids to remove.

        """
        if not np.any(to_remove):
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
        """Calculate penalties for assigning an instance to each centroid.

        Args:
            idx (int): Index of the instance to be predicted.
            iteration (int): Current iteration of the algorithm.

        Returns:
            numpy.ndarray: Penalties for assigning the instance to each centroid.

        """
        instance = self.X[idx]

        diff = self.centroids - np.repeat(
            instance[np.newaxis, :], self.n_clusters, axis=0
        )
        distances = self.distance(diff, axis=1).flatten()
        diff_allies = np.array(
            [self.diff_alliances(idx, c) for c in range(self.n_clusters)]
        )

        xi = self.x0 * (self.rate**iteration)
        return distances - (xi * diff_allies)

    def _fit(self):
        """Fits the RDPM model to the data."""
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

            logger.debug(
                f"Iteration {iteration} completed with clusters: {self.n_clusters}"
            )
            self.update()
