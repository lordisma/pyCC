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
from typing import Sequence

from constraints.matrix import ConstraintMatrix


class BaseEstimator(ABC, SklearnBaseEstimator):
    """
    Base class for estimators in the clustlib package.

    Notes
    -----
    All estimators should specify all the parameters that can be set at the class level in their ``__init__`` as
    explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    def fit(self, X, y=None):
        """Fit the model to the data.

        Parameters
        __________
        X: numpy.ndarray
            The data to cluster.
        y: numpy.ndarray, default=None
            Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns
        _______
        self
            The fitted estimator.
        """
        raise NotImplementedError

    def _update_centroids(self, X):
        """Update the centroids of the clusters.

        This method should be implemented by any class that inherits from it.

        Parameters
        __________
        X: numpy.ndarray
            The data to cluster.

        Returns
        _______
        numpy.ndarray
            The updated centroids.
        """
        raise NotImplementedError

    def _update_bounds(self, X):
        """Update the bounds of the clusters.
        
        This method should be implemented by any class that inherits from it.
        
        Parameters
        __________
        X: numpy.ndarray
            The data to cluster.
        """
        raise NotImplementedError

    def _update_labels(self, X):
        """Update the instances labels.
        
        This method should be implemented by any class that inherits from it.
        
        Parameters
        __________
        X: numpy.ndarray
            The data to cluster.
        """
        raise NotImplementedError

    def __calculate_centroids_distance(self):
        """Calculate the distance between centroids and other centroids.

        This method will update the __centroids_distance.
        """
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if i == j:
                    self.__centroids_distance[i, j] = np.inf # If it is the same centroid, the distance is infinite
                else:
                    self.__centroids_distance[j, i] = np.linalg.norm(
                        self.centroids[i] - self.centroids[j]
                    )
        
    def __check_cl_constraints(self, index):
        """Check the graph of constraints and return a list of valid centroids.

        This method should be overriden by when looking to do a soft clustering.
        since at the current moment it only takes into account the hard constraints.

        The lower bound of the instance will be set to infinity for invalid centroids.

        Parameters
        __________
        index: int
            Index of the instance to check.
        """
        cl_constraints = self.constraints.get_cl_constraints(index)
        cl = self.__get_centroids_from_constraints(cl_constraints)

        if cl.shape[0] == 0:
            self.__lower_bound[index, cl] = np.inf

    def __check_convergence(self):
        """Check convergence of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has converged, False otherwise.
        """
        # Check convergence
        return np.all(np.linalg.norm(self.__delta_centroid, axis=1) <= self.tol)

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

    def __get_valid_centroids(self, index):
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
        elif len(cl_prototypes) != 0:
            return np.delete(range_centroids, cl_prototypes) 
        else:
            return range_centroids

    def _initialize_bounds(self, dataset):
        """Initialize lower and upper bounds for each instance.

        This method will calculate the distance to each of the centroids in the cluster. 
        After that, it will assign the closest centroid to each instance and apply the constraints
        to make sure that the instances respect the limitations.

        In case of conflict, the instance that is closer to the centroid will be kept, and the other
        will be moved to the next closest centroid.

        FIXME: This method is not efficient and should be refactored.

        NOTE: This method applies the constraints in a soft manner. Which means that the instances
        might be missclassified after the initialization.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.

        Returns
        _______
        numpy.ndarray
            Lower bounds for each instance.
        numpy.ndarray
            Upper bounds for each instance.
        """
        lower_bounds = np.zeros((dataset.shape[0], self.n_clusters))
        upper_bounds = np.zeros((dataset.shape[0]))

        # Initialize lower and upper bounds
        for instance_index, instance in enumerate(dataset):
            for centroid_index, centroid in enumerate(self.centroids):
                lower_bounds[instance_index, centroid_index] = np.linalg.norm(
                    instance - centroid
                )

            # Get the closest centroid to the instance
            self._labels[instance_index] = lower_bounds[instance_index, :].argmin()

            upper_bounds[instance_index] = np.min(lower_bounds[instance_index, self._labels[instance_index]])

        # Apply the constraints to the newly created bounds and labels
        for instance_index in range(dataset.shape[0]):
            ml_constraints = self.constraints.get_ml_constraints(instance_index)
            cl_constraints = self.constraints.get_cl_constraints(instance_index)

            for ml_constraint in ml_constraints: # Soft ML constraints, it can be violated
                if self._labels[ml_constraint] != self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[ml_constraint]:
                        self._labels[instance_index] = self._labels[ml_constraint]
                        upper_bounds[instance_index] = lower_bounds[instance_index, self._labels[ml_constraint]]
                    else:
                        self._labels[ml_constraint] = self._labels[instance_index]
                        upper_bounds[ml_constraint] = lower_bounds[ml_constraint, self._labels[instance_index]]

            
            for cl_constraint in cl_constraints:
                if self._labels[cl_constraint] == self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[cl_constraint]:
                        lower_bounds[instance_index, self._labels[instance_index]] = np.inf
                        new_centroid = lower_bounds[instance_index, :].argmin()
                        self._labels[instance_index] = new_centroid
                        upper_bounds[instance_index] = lower_bounds[instance_index, new_centroid]
                    else:
                        lower_bounds[cl_constraint, self._labels[cl_constraint]] = np.inf
                        new_centroid = lower_bounds[cl_constraint, :].argmin()
                        self._labels[cl_constraint] = new_centroid
                        upper_bounds[cl_constraint] = lower_bounds[cl_constraint, new_centroid]


        return lower_bounds, upper_bounds

    def _initialize_centroids(self, dataset):
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

        return dataset[random_indices]