"""Model base
This module contains a BaseEstimator which provides the base's class for the rest of the estimator

Note
----
There is no intention to use this class directly, but to be inherited by other classes. Implementation is based on
scikit-learn's BaseEstimator in order to facilitate the integration with the library.

"""

from abc import ABC
from sklearn.base import ClusterMixin as SklearnBaseEstimator

import numpy as np

from .utils import ConstraintMatrix


class BaseEstimator(ABC, SklearnBaseEstimator):
    """
    Base class for estimators in the clustlib package.

    Notes
    -----
    All estimators should specify all the parameters that can be set at the class level in their ``__init__`` as
    explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    _labels: np.ndarray
    centroids: np.ndarray
    n_clusters: int
    tol: float
    constraints: ConstraintMatrix

    def fit(self, dataset, labels=None):
        """Fit the model to the data.

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        labels: numpy.ndarray, default=None
            Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns
        _______
        self
            The fitted estimator.
        """
        raise NotImplementedError

    def _update_centroids(self, dataset):
        """Update the centroids of the clusters.

        This method should be implemented by any class that inherits from it.

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.

        Returns
        _______
        numpy.ndarray
            The updated centroids.
        """
        raise NotImplementedError

    def _update_bounds(self, dataset):
        """Update the bounds of the clusters.

        This method should be implemented by any class that inherits from it.

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        raise NotImplementedError

    def _update_labels(self, dataset):
        """Update the instances labels.

        This method should be implemented by any class that inherits from it.

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        raise NotImplementedError

    def _calculate_centroids_distance(self):
        """Calculate the distance between centroids and other centroids.

        This method will update the _centroids_distance.
        """
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if i == j:
                    self._centroids_distance[
                        i, j
                    ] = np.inf  # If it is the same centroid, the distance is infinite
                else:
                    self._centroids_distance[j, i] = np.linalg.norm(
                        self.centroids[i] - self.centroids[j]
                    )

    def _check_convergence(self):
        """Check convergence of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has converged, False otherwise.
        """
        # Check convergence
        return np.all(np.linalg.norm(self._delta_centroid, axis=1) <= self.tol)

    def __get_centroids_from_constraints(self, constraints):
        """Get current class of a set of instances, this will be used to
        convert the list of instances into a list of centroids.

        Parameters
        __________
        constraints: list
            List of constraints.

        Returns
        _______
        numpy.ndarray
            Centroids from constraints.
        """
        # Get centroids from constraints
        # FIXME: This is assuming that the labels are correctly indexed on the moment of the clustering
        # so the conversion to centroids is possible.
        clusters = list(map(int, set(self._labels[list(constraints)])))
        return np.asarray(clusters, dtype=int)

    def _get_valid_centroids(self, index):
        """Check the graph of constrains and return a list of valid centroids.

        This method should be overriden by when looking to do a soft clustering.
        since at the current moment it only takes into account the hard constraints.

        Parameters
        __________
        index: int
            Index of the instance to check.

        Returns
        _______
        numpy.ndarray
            List of valid centroids.
        """
        ml_constraints = self.constraints.get_ml_constraints(index)
        if len(ml_constraints) == 0:
            ml_prototypes = []
        else:
            ml_prototypes = self.__get_centroids_from_constraints(ml_constraints)

        cl_constraints = self.constraints.get_cl_constraints(index)
        if len(cl_constraints) == 0:
            cl_prototypes = []
        else:
            cl_prototypes = self.__get_centroids_from_constraints(cl_constraints)

        range_centroids = np.arange(self.n_clusters)

        if len(ml_prototypes) != 0:
            return range_centroids[np.isin(range_centroids, ml_prototypes)]

        if len(cl_prototypes) != 0:
            return np.delete(range_centroids, cl_prototypes)

        return range_centroids

    def _initialize_centroids(self, dataset):
        """Initialize centroids randomly.

        Parameters
        __________
        dataset: numpy.ndarray
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

        return dataset[random_indices]
