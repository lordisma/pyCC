import numpy as np
from numpy import matlib
from .base import KMeansBaseEstimator
from typing import Sequence
from ..utils.matrix import ConstraintMatrix


class LCVQE(KMeansBaseEstimator):
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
                GMLV[c_i] = np.append(GMLV[c_i], j)
                GMLV[c_j] = np.append(GMLV[c_j], i)
            elif min_case == 1:
                # Change the cluster of the instance j
                self._labels[j] = c_i
            else:
                # Change the cluster of the instance i
                self._labels[i] = c_j

    def __check_cl_cases(self, dataset):
        """
        Get the distance of an instance to the closest centroid, and the distance to the other instances

        TODO: Implement the method to check the cl cases
        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        """
        for i, j in self.constraints.get_cl_cases():
            c_i, distance_c_i = self._get_closest_centroid(dataset[i])
            c_j, distance_c_j = self._get_closest_centroid(dataset[j])

            if c_i == c_j:
                continue

            if distance_c_j < distance_c_i:
                r_j = j
                closest_object = i

            distance_r_closet = self.__calculate_distance(dataset[r_j], MM_j)

            case_a = 0.5 * distance_c_i + 0.5 * distance_c_j + 0.5 * distance_r_closet
            case_b = 0.5 * distance_to_closet + 0.5 * distance_r_closet

            idx_min = np.argmin([case_a, case_b])

            if idx_min == 0:
                # Keep the instances in the same cluster
                GCLV[MM_j] = np.append(GCLV[MM_j], r_j)
            elif idx_min == 1:
                self._labels[closest_object] = c_i
            else:
                self._labels[i] = c_i
                self._labels[j] = other_cluster

    def __update_centroids(self, dataset):
        for c in range(K):
            members = np.where(idx == c)[0]
            coords_members = np.sum(X[members, :], 0)
            coords_GMLV = np.sum(X[np.array(GMLV[c], dtype=np.int), :], 0)
            coords_GCLV = np.sum(X[np.array(GCLV[c], dtype=np.int), :], 0)
            n_j = len(members) + 0.5 * len(GMLV[c]) + len(GCLV[c])
            if n_j == 0:
                n_j = 1
            centroids[c, :] = (coords_members + 0.5 * coords_GMLV + coords_GCLV) / n_j

    def __calculate_violation(self, dataset):
        for c in range(K):
            """
            Calculate the mean distance, the violation items and add that to the distances
            """
            lcvqe[c] = 0.5 * np.sum(distances[np.where(idx == c)[0], c], 0)
            sum_ML = 0
            sum_CL = 0

            for item_violated in GMLV[c]:
                sum_ML += distances[item_violated, c]

            for item_violated in GCLV[c]:
                sum_CL += distances[int(item_violated), int(c)]

            lcvqe[c] += 0.5 * sum_ML + 0.5 * sum_CL

        lcvqe = np.sum(lcvqe)

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

        self._lower_bounds, self._upper_bounds = self._initialize_bounds(dataset)

        for _ in range(self.max_iter):
            try:
                # Update centroids
                self._update_centroids(dataset)
            except ValueError:
                return self.centroids
            except Exception as error:
                raise error

            # Check convergence
            if self._check_convergence():
                break

        # Initialize clusters
        return self.centroids
