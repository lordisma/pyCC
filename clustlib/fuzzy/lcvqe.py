import numpy as np

from clustlib.model import BaseEstimator
from ..utils.distance import match_distance

import logging
logger = logging.getLogger(__name__)

class LCVQE(BaseEstimator):
    def __init__(
        self,
        constraints,
        n_clusters=8,
        init="random",
        distance="euclidean",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None
    ):
        self.n_clusters = n_clusters
        self.constraints = constraints
        self.distance = match_distance(distance)
        self.init = init

        self._delta_centroid = None
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None

    def _get_closest_centroid(self, instance):
        """
        Get the closest centroid to the instance.

        Parameters
        __________
        instance: numpy.ndarray
            The instance to find the closest centroid.

        Returns
        _______
        closest_centroid: int
            The index of the closest centroid.
        distance: float
            The distance to the closest centroid.
        """
        distances = np.linalg.norm(self.centroids - instance, axis=1)
        closest_centroid = np.argmin(distances)
        return closest_centroid, distances[closest_centroid]
    
    def get_ml_cases(self):
        """
        Get the must-link cases for the instance.

        Parameters
        __________
        instance: numpy.ndarray
            The instance to find the must-link cases.

        Returns
        _______
        ml_cases: list of tuples
            The must-link cases for the instance.
        """
        ml = np.copy(self.constraints)
        ml = ml - np.diag(np.diag(ml))  # Remove diagonal elements
        return np.argwhere(ml > 0)
    
    def get_cl_constraints(self):
        """
        Get the cannot-link constraints for the instance.

        Parameters
        __________
        instance: numpy.ndarray
            The instance to find the cannot-link constraints.

        Returns
        _______
        cl_constraints: list of tuples
            The cannot-link constraints for the instance.
        """
        cl = np.copy(self.constraints)
        cl = cl - np.diag(np.diag(cl))  # Remove diagonal elements
        return np.argwhere(cl < 0)

    def __check_ml_cases(self):
        """
        Get the distance of an instance to the closest centroid, and the distance to the other instances

        TODO: Implement the method to check the ml cases
        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        for i, j in self.get_ml_cases():
            c_i, distance_c_i = self._get_closest_centroid(self.X[i])
            c_j, distance_c_j = self._get_closest_centroid(self.X[j])

            if c_i != c_j:
                continue

            distance_i_cj = self.distance(self.X[i] - self.centroids[c_j])
            distance_j_ci = self.distance(self.X[j] - self.centroids[c_i])

            case_a = 0.5 * (distance_c_i + distance_c_j) + 0.25 * (
                distance_i_cj + distance_j_ci
            )
            case_b = 0.5 * distance_c_i + 0.5 * distance_c_j
            case_c = 0.5 * distance_i_cj + 0.5 * distance_j_ci

            min_case = np.argmin([case_a, case_b, case_c])

            if min_case == 0:
                # Keep the instances in the same cluster
                self.must_link_violations[c_i][j] = 1
                self.must_link_violations[c_j][i] = 1
            elif min_case == 1:
                # Change the cluster of the instance j
                self._labels[j] = c_i
            else:
                # Change the cluster of the instance i
                self._labels[i] = c_j

    def __check_cl_cases(self):
        """
        Get the distance of an instance to the closest centroid, and the distance to the other instances
        """
        for i, j in self.get_cl_constraints():
            cluster_i = self._labels[i]
            cluster_j = self._labels[j]

            if cluster_i != cluster_j:
                continue

            distances_i = self.distance(self.centroids - self.X[i], axis = 1)
            distances_j = self.distance(self.centroids - self.X[j], axis = 1)

            logging.deb

            vstack = np.array([distances_i, distances_j])
            mean_distances = np.argsort(np.mean(vstack, axis=1))
            
            logging.debug(f"Mean distances: {mean_distances}")
            mean_distances = np.setdiff1d(mean_distances, np.array([cluster_i, cluster_j]))

            farest_cluster = mean_distances[-1]
            intercluster_distances = np.linalg.norm(self.centroids - self.centroids[farest_cluster], axis=1)
            closest_cluster = np.setdiff1d(
                np.argsort(intercluster_distances), np.array([cluster_i, cluster_j, farest_cluster])
            )[0] 

            distance_r = np.linalg.norm(self.centroids[closest_cluster] - self.centroids[farest_cluster])
            if distances_i[cluster_i] > distances_j[cluster_i]:
                # If so we should move the instance i
                r_j = j 
            else:
                r_j = i

            A = 0.5 * distances_i[cluster_i] + 0.5 * distances_j[cluster_i] + 0.5 * distance_r
            B = 0.5 * distances_i[cluster_i] + 0.5 * distances_j[cluster_j]

            if A < B:
                self.cannot_link_violations[closest_cluster][r_j] = 1
                self._labels[i] = cluster_i
                self._labels[j] = cluster_i
            elif B < A:
                self._labels[i] = cluster_j if i != r_j else closest_cluster
                self._labels[j] = cluster_i if j != r_j else closest_cluster 
            else:
                self._labels[i] = cluster_i
                self._labels[j] = self._get_closest_centroid(self.X[j])[1]

    def _update(self):
        for c in range(self.n_clusters):
            members = np.where(self._labels == c)[0]
            coords_members = np.sum(self.X[members, :], 0)
            coords_GMLV = np.sum(self.X[np.array(self.must_link_violations[c], dtype=np.int), :], 0)
            coords_GCLV = np.sum(self.X[np.array(self.cannot_link_violations[c], dtype=np.int), :], 0)
            n_j = len(members) + 0.5 * len(self.cannot_link_violations[c]) + len(self.must_link_violations[c])
            if n_j == 0:
                n_j = 1
            self.centroids[c, :] = (coords_members + 0.5 * coords_GMLV + coords_GCLV) / n_j

    def _fit(self):
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
        self.must_link_violations = np.zeros((self.n_clusters, self.X.shape[0]))
        self.cannot_link_violations = np.zeros((self.n_clusters, self.X.shape[0]))
        iteration = 0

        logging.debug("Fitting LCVQE model")
        while not self.stop_criteria(iteration):
            logging.debug(f"Iteration {iteration}: Checking constraints")
            self.__check_ml_cases()
            self.__check_cl_cases()

            logging.debug(f"Iteration {iteration}: Updating centroids")
            self._update()

            logging.debug(f"Iteration {iteration}: Updating labels")
            iteration += 1

        # Initialize clusters
        return self.centroids
