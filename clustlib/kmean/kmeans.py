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
from sklearn.cluster._k_means_elkan import init_bounds_dense
from sklearn.cluster import KMeans
from typing import Sequence

from ..constraints.matrix import ConstraintMatrix

import rsinner as rs


class COPMeans(BaseEstimator):
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

        self.__delta_centroid = np.zeros(self.centroids.shape)

        # Get lower and upper bounds for each instance
        self.__lower_bounds, self.__upper_bounds = self.__init_bounds(dataset)

        for i in range(self.max_iter):
            # Assign instances to clusters
            self.__assign_instances(dataset)

            # Update centroids
            self.__update_centroids(dataset)

            # Check convergence
            if self.__check_convergence():
                break

        # Initialize clusters
        return self.centroids

    def __check_convergence(self):
        """Check convergence of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has converged, False otherwise.
        """

        # Check convergence
        return np.all(np.linalg.norm(self.__delta_centroid, axis=1) <= self.tol)

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
            current_class = self.__get_valid_centroid(instance_index)

            # Update upper bound for the instance
            self.__upper_bounds[instance_index] = np.min(
                self.__lower_bounds[instance_index, current_class]
            )

    def __get_valid_centroid(self, index):
        """Check the graph of constrains and return a list of valid centroids.

        # Assumtion: There is a Constraints Matrix specifing the valid centroids for each instance.

        Parameters
        __________
        index: int
            Index of the instance to check.
        """
        cl_constraints = list(
            self.constraints.get_cl_constraints(index)
        )  # Get a list of indexes of CL constraints
        if len(cl_constraints) != 0:
            cl_prototypes = list(
                map(int, set(self.__current_class[cl_constraints]))
            )  # Get a set of the current classes of the constraints
            self.__lower_bounds[
                index, cl_prototypes
            ] = np.inf  # Set the lower bounds of the invalid centroids to infinity

        ml_constraints = list(
            self.constraints.get_ml_constraints(index)
        )  # Get a list of indexes of ML constraints
        if len(ml_constraints) != 0:
            ml_prototypes = list(
                map(int, set(self.__current_class[ml_constraints]))
            )  # Get a set of the current classes of the constraints
            return self.__lower_bounds[
                index, ml_prototypes
            ].argmin()  # Return the index of the closest centroid
        else:
            return self.__lower_bounds[index].argmin()

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
            instances = dataset[self.__current_class == centroid_index]

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
                self.__delta_centroid[self.__current_class[instance_index]]
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

        self.__current_class = np.zeros((dataset.shape[0]), dtype=int)

        for instance_index, instance in enumerate(dataset):
            for centroid_index, centroid in enumerate(self.centroids):
                lower_bounds[instance_index, centroid_index] = np.linalg.norm(
                    instance - centroid
                )

            # Get the closest centroid to the instance
            self.__current_class[instance_index] = lower_bounds[
                instance_index, :
            ].argmin()

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

    def fit_predict(self, dataset):
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.

        Returns
        _______
        numpy.ndarray
            Index of the cluster each sample belongs to.
        """
        self.fit(dataset)
        return self.predict(dataset)
