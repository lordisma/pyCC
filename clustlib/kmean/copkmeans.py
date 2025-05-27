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

from typing import Sequence
import numpy as np
from .. import BaseEstimator

from ..utils.matrix import ConstraintMatrix
from ..utils.simpleconstraints import SimpleConstraints
from ..utils.distance import match_distance


# FIXME: Devuelve un stimator, ajustarse a sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit
class COPKMeans(BaseEstimator):
    # pylint: disable=too-many-instance-attributes
    """COPKMeans

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
        constraints: Sequence[Sequence],
        n_clusters=8,
        init="random",
        distance="euclidean",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
    ):
        self._delta_centroid = None
        self.n_clusters = n_clusters
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.constraints = constraints
        self.distance = match_distance(distance)

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
        self._labels = np.zeros(dataset.shape[0], dtype=int)
        self.X = np.copy(dataset)

        self._lower_bounds, self._upper_bounds = self._initialize_bounds(dataset)

        iteration = 0

        while self._convergence(iteration):
            iteration += 1
            try:
                # Update centroids
                self.update()
            except ValueError:
                return self.centroids
            except Exception as error:
                raise error

        # Initialize clusters
        return self.centroids
    
    def _get_centroids(self, idx):
        """Get the valid centroids for the instance.

        This method checks the constraints for the instance and returns the valid centroids.
        Parameters
        __________
        idx: int
            The index of the instance to check.
        Returns
        _______
        valid_centroids: numpy.ndarray
            The valid centroids for the instance.
        """
        ml = self.constraints.get_ml_constraints(idx)
        cl = self.constraints.get_cl_constraints(idx)

        if ml is not None and len(ml) > 0:
            return ml
        
        if cl is not None and len(cl) > 0:
            valid_centroids = np.delete(np.arange(self.n_clusters), cl)
            return valid_centroids
        
        return np.arange(self.n_clusters)

    def _update_label(self, idx):
        """Update the instances labels.

        This method follows the Elkan's algorithm to update the labels of the instances.

        Parameters
        __________
        idx: int
            The index of the instance to update.
        """
        instance = self.X[idx]
        valid_centroids = self._get_centroids(idx)

        if len(valid_centroids) == 0:
            raise ValueError("Invalid set of centroids")
        
        self._lower_bounds[idx, np.isin(np.arange(self.n_clusters), valid_centroids, invert=True)] = np.inf

        if len(valid_centroids) == 1:
            centroid = valid_centroids[0]
            distance = self.distance(instance - self.centroids[centroid])
            self._labels[idx] = centroid
            self._upper_bounds[idx] = distance
            self._lower_bounds[idx, centroid] = distance
            return

        current_distance = self._upper_bounds[idx]
        current_centroid = self._labels[idx]
        closest_centroid = self._centroids_distance[current_centroid, :].argmin()
        min_distance = self._centroids_distance[current_centroid, closest_centroid]

        if current_centroid == closest_centroid:
            # The centroid is choosing itself again, we should update the instance
            # to keep him off
            self._centroids_distance[current_centroid, closest_centroid] = np.inf
            closest_centroid = self._centroids_distance[current_centroid, :].argmin()
            min_distance = self._centroids_distance[current_centroid, closest_centroid]

        if current_distance > 0.5 * min_distance:
            # Set the instance to the current centroid
            for centroid_index in valid_centroids:
                candidate = self.centroids[centroid_index]
                current = self.centroids[self._labels[idx]]

                # Check if the current distance must be updated
                if self.__should_check_centroid(self._labels[idx], centroid_index, idx):
                    distance_to_candidate = self.distance(instance - candidate)
                    distance_to_current_centroid = self.distance(instance - current)

                    if distance_to_candidate < distance_to_current_centroid:
                        self._labels[idx] = centroid_index
                        self._upper_bounds[idx] = distance_to_candidate

                    self._lower_bounds[idx, centroid_index] = distance_to_candidate
                    self._lower_bounds[idx, current_centroid] = current_distance

    def _update(self):
        """Get the instances belonging to each cluster and update the centroids,
        and upper and lower bounds.

        FIXME: This method should be split into smaller methods

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """
        for label, _ in enumerate(self.centroids):
            self.centroids[label] = self.X[np.where(self._labels == label)].mean(axis=0)
            self._centroids_distance[:, label] = self.distance(
                self.centroids - np.tile(self.centroids[label], self.centroids.shape[0]), axis=1
            )

        self._update_bounds()

    def _update_bounds(self):
        """Update lower and upper bounds for each instance.

        Parameters
        __________
        X: numpy.ndarray
            Training instances to cluster.
        """

        for centroids_index, _ in enumerate(self.centroids):
            members = np.where(self._labels == centroids_index)[0]
            self._upper_bounds[members] += self.distance(self.__delta[centroids_index])

            self._lower_bounds[members, :] -= np.tile(
                self.__delta, (len(members), 1)
            )

        for idx in len(self.X):
            self._update_label(idx)

    def __should_check_centroid(self, centroid_index, candidate_centroid, idx):
        """Check if the candidate centroid is a valid option for the instance.

        Parameters
        __________
        centroid_index: int
            The current centroid index.
        candidate_centroid: int
            The candidate centroid index.
        idx: int
            The instance index.

        Returns
        _______
        bool
            True if the candidate centroid is a valid option for the instance, False otherwise.
        """
        half_distance = (
            0.5 * self._centroids_distance[centroid_index, candidate_centroid]
        )
        return (
            self._upper_bounds[idx] > self._lower_bounds[idx, candidate_centroid]
        ) and (self._upper_bounds[idx] > half_distance)
