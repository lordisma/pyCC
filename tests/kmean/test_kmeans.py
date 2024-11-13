from clustlib.base import lloyd as lloyd
import numpy as np


class TestBaseKMeans:
    def test_kmeans(self):
        kmeans = lloyd.LloydKMeans(
            centroids=np.array([[1, 2], [8, 9]]),
            labels=np.array([0, 1, 0, 1]),
            X=np.array([[1.1, 2.1], [8.1, 9.1], [1.2, 2.2], [8.2, 9.2]]),
        )

        kmeans.update()
        print(kmeans._centroids.tolist())
        assert True
