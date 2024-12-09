from .kmean import KMeans
import numpy as np


class ElkanKMeans(KMeans):
    _lower_bounds: np.ndarray
    _upper_bounds: np.ndarray

    _delta_centroid: np.ndarray
    _distance: np.ndarray

    _tolerance: float

    def _update_bounds(self):
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
        dataset: numpy.ndarray
            Training instances to cluster.

        Returns
        _______
        numpy.ndarray
            Lower bounds for each instance.
        numpy.ndarray
            Upper bounds for each instance.
        """
        self._update_distance()

        self._lower_bounds = self._distances.copy()
        self._upper_bounds = np.min(self._lower_bounds, axis=1)

        self._labels = np.argmin(self._lower_bounds, axis=1)

    def _update(self):
        """
        Update the centroids and the bounds for each instance in the dataset
        """
        previous_centroids = self._centroids.copy()

        for c in range(self._centroids.shape[0]):
            previous_centroid = self._centroids[c]
            self._centroids[c, :] = np.mean(self.X[self._labels == c], axis=0)

            self._delta_centroid[c] = np.linalg.norm(
                self._centroids[c], previous_centroid
            )

        if np.any(
            abs(np.sum((self._delta_centroid / previous_centroids) * 100))
            < self._tolerance
        ):
            self._update_bounds()
            return

        for i in range(self._centroids.shape[0]):
            self._lower_bounds[:, i] -= self._delta_centroid[i]
            self._upper_bounds[np.where(self._labels == i)] += self._delta_centroid[i]

    def fit(self):
        while self._convergence():
            self._update()

            _distance_between_centroids = np.linalg.norm(
                self._centroids - self._centroids[:, None], axis=-1
            )
            np.fill_diagonal(_distance_between_centroids, np.inf)

            for i in range(self._centroids.shape[0]):
                half_distance_to_closest = (
                    np.min(_distance_between_centroids[i, :]) * 0.5
                )

                greater_than = np.where(self._upper_bounds > half_distance_to_closest)
                label_is = np.where(self._labels == i)
                points_to_consider = np.intersect1d(
                    greater_than, label_is, assume_unique=True
                )

                if points_to_consider.shape[0] == 0:
                    self._lower_bounds[label_is, i] -= self._delta_centroid[i]
                    self._upper_bounds[label_is] += self._delta_centroid[i]
                else:
                    for idx in points_to_consider:
                        current_distance = self._upper_bounds[idx]

                        distance_is_bigger = np.where(
                            self._lower_bounds[idx, :] < current_distance
                        )
                        inter_centroid_distance = np.where(
                            (_distance_between_centroids[i, :] * 0.5) < current_distance
                        )

                        target_centroids = np.intersect1d(
                            distance_is_bigger,
                            inter_centroid_distance,
                            assume_unique=True,
                        )

                        if target_centroids.shape[0] == 0:
                            self._lower_bounds[idx, :] -= self._delta_centroid[i]
                            self._upper_bounds[idx] += self._delta_centroid[i]
                        else:
                            self._lower_bounds[idx, :] = np.linalg.norm(
                                self.X[idx] - self._centroids, axis=1
                            )
                            if np.any(self._lower_bounds[idx, :] < current_distance):
                                self._labels[idx] = np.argmin(
                                    self._lower_bounds[idx, :]
                                )
                                self._upper_bounds[idx] = np.min(
                                    self._lower_bounds[idx, :]
                                )

                    rest_points = np.setdiff1d(
                        label_is, points_to_consider, assume_unique=True
                    )
                    self._lower_bounds[rest_points, :] -= self._delta_centroid[i]
                    self._upper_bounds[rest_points] += self._delta_centroid[i]
