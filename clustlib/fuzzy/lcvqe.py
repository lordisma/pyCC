import numpy as np
from ..base import BaseEstimator
from typing import Sequence
from ..utils.matrix import ConstraintMatrix


class LCVQE(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
    ):
        self._delta_centroid = None
        self.n_clusters = n_clusters
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.constraints = ConstraintMatrix(constraints)

    def __check_ml_cases(self, dataset):
        """
        Get the distance of an instance to the closest centroid, and the distance to the other instances

        TODO: Implement the method to check the ml cases
        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        for i, j in self.constraints.get_ml_cases():
            c_i, distance_c_i = self._get_closest_centroid(dataset[i])
            c_j, distance_c_j = self._get_closest_centroid(dataset[j])

            if c_i != c_j:
                continue

            distance_i_cj = self.__calculate_distance(dataset[i], self.centroids[c_j])
            distance_j_ci = self.__calculate_distance(dataset[j], self.centroids[c_i])

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

    def __check_cl_cases(self, dataset):
        """
        Get the distance of an instance to the closest centroid, and the distance to the other instances

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        for i, j in self.constraints.get_cl_constraints():
            cluster_i = self._labels[i]
            cluster_j = self._labels[j]

            if cluster_i != cluster_j:
                continue

            distances_i = np.linalg.norm(self.centroids - self.X[i], axis=1)
            distances_j = np.linalg.norm(self.centroids - self.X[j], axis=1)

            mean_distances = np.argsort(np.mean(np.hstack((distances_i, distances_j)), axis=1))
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

    def __update_centroids(self, dataset):
        for c in range(self.n_clusters):
            members = np.where(self._labels == c)[0]
            coords_members = np.sum(dataset[members, :], 0)
            coords_GMLV = np.sum(dataset[np.array(self.must_link_violations[c], dtype=np.int), :], 0)
            coords_GCLV = np.sum(dataset[np.array(self.cannot_link_violations[c], dtype=np.int), :], 0)
            n_j = len(members) + 0.5 * len(self.cannot_link_violations[c]) + len(self.must_link_violations[c])
            if n_j == 0:
                n_j = 1
            self.centroids[c, :] = (coords_members + 0.5 * coords_GMLV + coords_GCLV) / n_j

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
        self.centroids = self._initialize_centroids(dataset)
        self._delta_centroid = np.zeros(self.centroids.shape)
        self._labels = np.zeros(dataset.shape[0], dtype=int)

        self.must_link_violations = np.zeros((self.n_clusters, dataset.shape[0]))
        self.cannot_link_violations = np.zeros((self.n_clusters, dataset.shape[0]))

        for _ in range(self.max_iter):
            try:
                self.__check_ml_cases(dataset)
                self.__check_cl_cases(dataset)

                # Update centroids
                self.__update_centroids(dataset)
            except ValueError:
                return self.centroids
            except Exception as error:
                raise error

            # Check convergence
            if self._check_convergence():
                break

        # Initialize clusters
        return self.centroids
