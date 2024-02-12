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
from sklearn.cluster import KMeans
from typing import Sequence

from ..constraints.matrix import ConstraintMatrix


# FIXME: Fix the conjunto vacio, CL 3 instancias y dos clusters
# FIXME: Esto es el Soft-COP-KMeans
# FIXME: Devuelve un stimator, ajustarse a sklearn https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit
# TODO: Crear una clase base
class BaseKMeans(BaseEstimator):
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

    __lower_bounds: np.ndarray
    __upper_bounds: np.ndarray
    __delta_centroid: np.ndarray
    _labels: np.ndarray
    __cl_cluster: np.ndarray
    __ml_cluster: np.ndarray

    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.constraints = ConstraintMatrix(constraints)

    def fit(self, dataset):
        """Compute k-means clustering.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """
        # Initialize centroids
        if self.init == "random":
            self.centroids = self.__init_random(dataset)
        elif self.init == "custom":
            self.centroids = self.__init_custom()
        else:
            raise ValueError(f"Invalid init method: '{self.init}'")

        # FIXME: Calcular diametro del dataset (sive tanto para la normaizacion como sin ella)

        self.__delta_centroid = np.zeros(self.centroids.shape)

        # Get lower and upper bounds for each instance
        self.__lower_bounds, self.__upper_bounds = self.__init_bounds(dataset)

        # Generate the clusters and constraints matrix
        self.__generate_clusters_constraints(dataset)

        for i in range(self.max_iter):
            # Assign instances to clusters
            self.__assign_instances(dataset)
            try:
                # Update centroids
                self.__update_centroids(dataset)
            except ValueError as e:
                print("ERROR: Unsustainable solution found")
                self._labels = np.zeros(len(dataset))
                return self.centroids
            except Exception as e:
                raise e

            # Check convergence
            if self.__check_convergence():
                break

        # Initialize clusters
        return self.centroids

    def __generate_clusters_constraints(self, dataset):
        """Generate clusters constraints.

        TODO: Fix the docstring
        Parameters
        __________
        dataset: numpy.ndarray
            Training instances to cluster.
        """
        for index in range(len(dataset)):
            cl_constraints = list(self.constraints.get_cl_constraints(index))
            if len(cl_constraints) != 0:
                cl_prototypes = self.__get_centroids_from_constraints(cl_constraints)
                self.__check_invalid_set(cl_prototypes)
                self.__cl_cluster[index] = cl_prototypes

            ml_constraints = list(self.constraints.get_ml_constraints(index))
            if len(ml_constraints) != 0:
                self.__ml_cluster[index] = self.__get_centroids_from_constraints(
                    ml_constraints
                )

    def __check_convergence(self):
        """Check convergence of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has converged, False otherwise.
        """
        # Check convergence
        return np.all(np.linalg.norm(self.__delta_centroid, axis=1) <= self.tol)

    def __check_invalid_set(self, cl_centroids):
        """Check if the set of centroids is invalid. And raise an exception if it is

        Parameters
        __________
        cl_centroids: numpy.ndarray
            Centroids of the current class.

        Returns
        _______
        bool
            True if the set of centroids is invalid, False otherwise.
        """
        # Check if the set of centroids is invalid
        if len(cl_centroids) == self.n_clusters:
            raise ValueError("Invalid set of centroids")

    def __assign_instances(self, dataset):
        """Assign instances to clusters.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """
        # Assign each instance to the closest centroid
        for instance_index, _ in enumerate(dataset):
            # Get the closest centroid to the instance
            current_class = self.__get_current_class(instance_index)

            # Update upper bound for the instance
            self.__upper_bounds[instance_index] = np.min(
                self.__lower_bounds[instance_index, current_class]
            )

    def __get_current_class(self, index):
        """Get current class of a set of instances, this will be used to
        convert the list of instances into a list of centroids.

        Parameters
        __________
        index: int
            Index of the instance to check.

        Returns
        _______
        numpy.ndarray
            Centroids from constraints.
        """
        # Get current class of the instance
        return self.__get_valid_centroids(index).argmin()

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
        return list(map(int, set(self._labels[constraints])))

    def __get_valid_centroids(self, index):
        """Check the graph of constrains and return a list of valid centroids.

        # Assumtion: There is a Constraints Matrix specifing the valid centroids for each instance.

        Parameters
        __________
        index: int
            Index of the instance to check.

        Returns
        _______
        numpy.ndarray
            List of valid centroids.
        """
        ml_prototypes = self.__ml_cluster[index]

        if len(ml_prototypes) != 0:
            return self.__lower_bounds[index, ml_prototypes]
        else:
            return self.__lower_bounds[index]

    def __update_centroids(self, dataset):
        """Update centroids.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """
        # Update centroids
        for centroid_index, _ in enumerate(self.centroids):
            # Get instances belonging to the current cluster
            instances = dataset[self._labels == centroid_index]

            # Save the old value of the centroid to calculate the delta
            old_centroid = self.centroids[centroid_index]

            # Update centroid
            self.centroids[centroid_index] = instances.mean(axis=0)

            # Calculate the delta
            self.__delta_centroid[centroid_index] = (
                self.centroids[centroid_index] - old_centroid
            )

        # Update lower and upper bounds
        self.__update_bounds(dataset)

    def __update_bounds(self, dataset):
        """Update lower and upper bounds for each instance.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """

        for instance_index, _ in enumerate(dataset):
            self.__lower_bounds[instance_index, :] -= np.linalg.norm(
                self.__delta_centroid, axis=1
            )  # May be wrong
            self.__upper_bounds[instance_index] += np.linalg.norm(
                self.__delta_centroid[self._labels[instance_index]]
            )

    def __init_bounds(self, dataset):
        """Initialize lower and upper bounds for each instance.

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

        self._labels = np.zeros((dataset.shape[0]), dtype=int)

        for instance_index, instance in enumerate(dataset):
            for centroid_index, centroid in enumerate(self.centroids):
                lower_bounds[instance_index, centroid_index] = np.linalg.norm(
                    instance - centroid
                )

            # Get the closest centroid to the instance
            self._labels[instance_index] = lower_bounds[instance_index, :].argmin()

            upper_bounds[instance_index] = np.min(lower_bounds[instance_index, :])

        return lower_bounds, upper_bounds

    def __init_random(self, dataset):
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

    def __init_custom(self):
        """Initialize centroids using custom_initial_centroids.

        Returns
        _______
        numpy.ndarray
            Set of centroids
        """
        # FIXME: Custom should be a funcition that allow
        return self.custom_initial_centroids
