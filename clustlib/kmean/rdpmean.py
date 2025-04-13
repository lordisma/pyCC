import numpy as np
from scipy import spatial as sdist
from typing import Tuple

from ..model import BaseEstimator


class RDPM(BaseEstimator):
    def initialize(
        self,
        n_clusters,
        constraints,
        lamb=0.001,
        max_iter=300,
        limit = 0.005
    ):
        self.n_clusters = n_clusters
        self.constraints = constraints
        self.lamb = lamb
        self.__centroids = self.normalize_centroids()
        self.max_iter = max_iter
        self.__prev_centroids: np.ndarray = np.zeros(
            (self.n_clusters, self.X.shape[0])
        )
        self.__min_tolrance = 0.0001
        self.limit = limit

    def normalize_centroids(self) -> np.ndarray:
        np.random.uniform(
            np.min(self.X, 0), np.max(self.X, 0), (self.n_clusters, self.X.shape[1])
        )

    def stop_criteria(self, iteration) -> bool:
        differences = np.array(
            list(
                map(
                    lambda pair: np.linalg.norm(pair[0] - pair[1]),
                    zip(self.__centroids, self.__prev_centroids),
                )
            )
        )

        return (iteration >= self.max_iter) or np.all(differences < self.__min_tolrance)

    def __check_alliances(self, d, c) -> Tuple[np.ndarray, np.ndarray]:
        friends = np.where(self.constraints[:, d] == 1)[0]
        strangers = np.where(self.constraints[:, d] == -1)[0]

        return self.X[friends], self.X[strangers]

    def __update_rate(self, iteration) -> float:
        return self.xi_0 * (self.xi_rate**iteration)

    def distance(self, d, c) -> float:
        return sdist.distance.euclidean(self.X[d, :], self.__centroids[c, :])

    def update_centroid(self, c) -> np.ndarray:
        if self.X[self.labels == c].shape[0] > 0:
            return np.mean(self.X[self.labels == c], axis=0)
        else:
            return np.zeros(self.X.shape[1])

    def __update_centroids(self):
        self.__centroids = np.array(
            [self.update_centroid(c) for c in range(self.n_clusters)]
        )
        
        for c in range(self.n_clusters):
            if np.all(self.labels != c):
                self.n_clusters -= 1
                self.__centroids = np.delete(self.__centroids, c, axis=0)

    def fit(self, dataset: np.ndarray, labels: np.array = None):
        self.X = dataset

        if labels:
            self._labels = labels
        else: 
            self._labels = np.random.randint(0, self.n_clusters, self.X.shape[0])

        iteration = 0
        while not self.stop_criteria(iteration):
            iteration += 1
            xi = self.__update_rate(iteration)

            for d in np.random.permutation(self.X.shape[0]):
                min_diff = float("inf")
                current_label = -1

                for c in range(self.n_clusters):
                    if iteration > 1:
                        friends, strangers = self.__check_alliances(d, c)

                    difference = len(friends) - len(strangers)
                    distance = self.distance(d, c)

                    penalty = distance - (xi * difference)

                    if min_diff > penalty:
                        min_diff = penalty
                        current_label = c

                # Check if the distance to the nearest centroid is greater than a limit, if so create a new cluster
                if min_diff >= self.limit:
                    self.n_clusters += 1
                    self.__centroids = np.vstack((self.__centroids, self.X[d, :]))
                    current_label = self.__centroids.shape[0] - 1

                self.labels[d] = current_label

            self.__prev_centroids = self.__centroids
            self.__update_centroids()
