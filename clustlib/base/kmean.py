import numpy as np


class KMeans:
    _centroids: np.ndarray
    _labels: np.ndarray
    _distances: np.ndarray
    X: np.ndarray

    def __init__(self, centroids: np.ndarray, labels: np.ndarray, X: np.ndarray):
        self._centroids = centroids
        self._labels = labels
        self.X = X

    def _update_distance(self):
        numb_obj = self.X.shape[0]

        for c in range(self._centroids.shape[0]):
            diff_to_centroid = self.X - np.tile(self._centroids[c, :], (numb_obj, 1))
            self._distances[:, c] = np.sum(np.power(diff_to_centroid, 2), 1).T

    def update(self):
        raise NotImplementedError
