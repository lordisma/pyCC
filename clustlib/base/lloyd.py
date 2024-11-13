import numpy as np
from .kmean import KMeans


class LloydKMeans(KMeans):
    def __iter__(self):
        return self

    def __next__(self):
        self.update()
        return self._centroids, self._labels

    def update(self):
        for c in range(self._centroids.shape[0]):
            self._centroids[c, :] = np.mean(self.X[self._labels == c], axis=0)

        self._update_distance()
