# -*- coding: utf-8 -*-
from typing import Sequence
import numpy as np
from .. import BaseEstimator

from ..utils.simpleconstraints import SimpleConstraints
from ..utils.distance import match_distance
from time import time

from logging import getLogger

logger = getLogger(__name__)

class COPKMeans(BaseEstimator):
    """KMeans estimator is a clustering algorithm that aims to partition n observations into k clusters in which each
    observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a
    partitioning of the data space into Voronoi cells.

    Attributes:
        n_clusters (:obj:`int`, optional): The number of clusters to form as well as the number of centroids to generate.
        init (:obj:`str`, optional): Method for initialization, defaults to 'random' choose k observations (rows) at 
            random from data for the initial centroids. 'custom' use custom_initial_centroids as initial centroids.
        max_iter (:obj:`int`, optional): Maximum number of iterations of the k-means algorithm for a single run.
        tol (:obj:`float`, optional): Relative tolerance with regards to Frobenius norm of the difference in the cluster
            centers of two consecutive iterations to declare convergence.
        custom_initial_centroids (:obj:`numpy.ndarray`, optional): Custom initial centroids to be used in the 
            initialization. Only used if init='custom'.
    """

    def __init__(
        self,
        constraints: Sequence[Sequence],
        n_clusters=8,
        init="random",
        distance="euclidean",
        max_iter=300,
        tol=1e-10,
        custom_initial_centroids=None,
    ):
        self.n_clusters = n_clusters
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.distance = match_distance(distance)
        self._labels = None
        self.constraints = SimpleConstraints(constraints)

    def _initialize_bounds(self):
        """This method will calculate the distance to each of the centroids in the cluster.
        After that, it will assign the closest centroid to each instance and apply the constraints
        to make sure that the instances respect the limitations.

        In case of conflict, the instance that is closer to the centroid will be kept, and the other
        will be moved to the next closest centroid.

        Note: 
            This method applies the constraints in a soft manner. Which means that the instances
            might be missclassified after the initialization.

        Args:
            dataset (ndarray): Training instances to cluster.

        Returns:
            numpy.ndarray: Lower bounds for each instance.
            numpy.ndarray: Upper bounds for each instance.
        """
        logger.debug("Initializing bounds for COPKMeans")
        lower_bounds = np.zeros((self.X.shape[0], self.n_clusters))
        upper_bounds = np.zeros((self.X.shape[0]))

        # Initialize lower and upper bounds
        for instance_index, instance in enumerate(self.X):
            for centroid_index, centroid in enumerate(self.centroids):
                lower_bounds[instance_index, centroid_index] = np.linalg.norm(
                    instance - centroid
                )

            # Get the closest centroid to the instance
            self._labels[instance_index] = lower_bounds[instance_index, :].argmin()

            upper_bounds[instance_index] = np.min(
                lower_bounds[instance_index, self._labels[instance_index]]
            )

        # Apply the constraints to the newly created bounds and labels
        for instance_index in range(self.X.shape[0]):
            constraints = self.constraints[instance_index]
            ml_constraints = np.argwhere(constraints > 0).flatten()
            cl_constraints = np.argwhere(constraints < 0).flatten()

            for (
                ml_constraint
            ) in ml_constraints:  # Soft ML constraints, it can be violated
                if self._labels[ml_constraint] != self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[ml_constraint]:
                        self._labels[instance_index] = self._labels[ml_constraint]
                        upper_bounds[instance_index] = lower_bounds[
                            instance_index, self._labels[ml_constraint]
                        ]
                    else:
                        self._labels[ml_constraint] = self._labels[instance_index]
                        upper_bounds[ml_constraint] = lower_bounds[
                            ml_constraint, self._labels[instance_index]
                        ]

            for cl_constraint in cl_constraints:
                if self._labels[cl_constraint] == self._labels[instance_index]:
                    if upper_bounds[instance_index] > upper_bounds[cl_constraint]:
                        lower_bounds[
                            instance_index, self._labels[instance_index]
                        ] = np.inf
                        new_centroid = lower_bounds[instance_index, :].argmin()
                        self._labels[instance_index] = new_centroid
                        upper_bounds[instance_index] = lower_bounds[
                            instance_index, new_centroid
                        ]
                    else:
                        lower_bounds[
                            cl_constraint, self._labels[cl_constraint]
                        ] = np.inf
                        new_centroid = lower_bounds[cl_constraint, :].argmin()
                        self._labels[cl_constraint] = new_centroid
                        upper_bounds[cl_constraint] = lower_bounds[
                            cl_constraint, new_centroid
                        ]

        return lower_bounds, upper_bounds

    def _fit(self):
        """Fit the model to the data.
        """
        self._lower_bounds, self._upper_bounds = self._initialize_bounds()

        logger.debug("Starting the iterations for COPKMeans")

        start = time()
        iteration = 0
        while not self.stop_criteria(iteration):
            iteration += 1
            logger.debug(
                f"Iteration {iteration} of {self.max_iter} for COPKMeans"
            )

            self.update()

        logger.debug(f"COPKMeans finished after {time() - start:.2f}")
    
    def _get_centroids(self, idx):
        """Get the valid centroids for the instance.

        This method checks the constraints for the instance and returns the valid centroids.
        
        Args:
            idx(int): The index of the instance to check.
        
        Returns:
            numpy.ndarray: The valid centroids for the instance.
        """
        constraints = self.constraints[idx]
        ml = np.argwhere(constraints > 0).flatten()
        cl = np.argwhere(constraints < 0).flatten()

        if ml is not None and len(ml) > 0:
            logger.debug(
                f"Instance {idx} has must-be-link constraints: {len(ml)}"
            )
            labels = np.unique(self._labels[ml])
            return labels
        
        if cl is not None and len(cl) > 0:
            labels = np.unique(self._labels[cl])
            valid_centroids = np.delete(
                np.arange(self.n_clusters), 
                labels
            )
            return valid_centroids
        
        return np.arange(self.n_clusters)

    def _update_label(self, idx):
        """Update the instances labels.

        This method follows the Elkan's algorithm to update the labels of the instances.

        Args:
            idx (int): The index of the instance to update.
        """
        instance = np.copy(self.X[idx])
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
            logger.debug(f"Instance {idx} is too far from the current centroid {current_centroid}, ")
            for centroid_index in valid_centroids:
                logger.debug(f"Checking candidate {centroid_index} for instance {idx}")
                candidate = self.centroids[centroid_index]
                current = self.centroids[self._labels[idx]]

                # Check if the current distance must be updated
                if self.__should_check_centroid(self._labels[idx], centroid_index, idx):
                    logger.debug(
                        f"Checking instance {idx} with centroid {centroid_index} "
                        f"and current centroid {self._labels[idx]}"
                    )
                    # Calculate the distance to the candidate centroid and the current centroid
                    distance_to_candidate = self.distance(instance - candidate)
                    distance_to_current_centroid = self.distance(instance - current)

                    if distance_to_candidate < distance_to_current_centroid:
                        logger.debug(
                            f"Updating instance {idx} from centroid {self._labels[idx]} to {centroid_index}")
                        self._labels[idx] = centroid_index
                        self._upper_bounds[idx] = distance_to_candidate

                    self._lower_bounds[idx, centroid_index] = distance_to_candidate
                    self._lower_bounds[idx, current_centroid] = current_distance

    def _update(self):
        """Get the instances belonging to each cluster and update the centroids,
        and upper and lower bounds.

        Args:
            X (numpy.ndarray): Training instances to cluster.
        """
        old = np.copy(self.centroids)

        for label in range(self.n_clusters):
            self.centroids[label] = np.copy(self.X[np.where(self._labels == label)].mean(axis=0))
            logger.debug(f"Centroid {label} updated to {self.centroids[label]}")

            intracentroid_diff = self.centroids - self.centroids[label]
            self._centroids_distance[:, label] = self.distance(intracentroid_diff, axis=1)

        if self._delta is None:
            self._delta = self.calculte_delta(old)

        self._update_bounds()

    def _update_bounds(self):
        """Update lower and upper bounds for each instance.

        Args:
            X (numpy.ndarray): Training instances to cluster.
        """

        distance = self.distance(self._delta, axis = 1)

        logger.debug(f"Updating bounds with delta: {distance}")

        for centroids_index in range(self.n_clusters):
            members = np.where(self._labels == centroids_index)
            self._upper_bounds[members] += distance[centroids_index]
            self._lower_bounds[members, :] -= distance[centroids_index]

        for idx in range(self.X.shape[0]):
            self._update_label(idx)

    def __should_check_centroid(self, centroid_index, candidate_centroid, idx):
        """Check if the candidate centroid is a valid option for the instance.

        Args:
            centroid_index (int):The current centroid index.
            candidate_centroid (int): The candidate centroid index.
            idx (int): The instance index.

        Returns:
            boolean: True if the candidate centroid is a valid option for the instance, False otherwise.
        """
        half_distance = (
            0.5 * self._centroids_distance[centroid_index, candidate_centroid]
        )
        return (
            self._upper_bounds[idx] > self._lower_bounds[idx, candidate_centroid]
        ) and (self._upper_bounds[idx] > half_distance)
