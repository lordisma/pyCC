import numpy as np
import math as m
import scipy.special as sp
import scipy.stats as st

import itertools as it


class TVClust:
    def __init__(self, num_clusters, X, constraints, max_iter=300, tol=0.0001):
        self.num_clusters = num_clusters
        self.X = X
        self.constraints = constraints
        self.max_iter = max_iter
        self.tol = tol
        self.__centroids = self.normalize_centroids()
        self.labels = np.random.randint(0, self.num_clusters, self.X.shape[0])
        # Probability of keeping the same cluster than another may-link
        self.__p = 0.9
        # Probability of creating a may-not-link false
        self.__q = 0.1
        # Parameter of concentation
        self.__alpha = 1.0

    def normalize_centroids(self):
        return np.random.uniform(
            np.min(self.X, 0), np.max(self.X, 0), (self.num_clusters, self.X.shape[1])
        )

    def stop_criteria(self, iteration):
        differences = np.array(
            list(
                map(
                    lambda pair: np.linalg.norm(pair[0] - pair[1]),
                    zip(self.__centroids, self.__prev_centroids),
                )
            )
        )

        return (iteration >= self.max_iter) or np.all(differences < self.tol)

    def should_create_new_cluster(self, i):
        new_cluster_mean = np.mean(self.X, 0)
        new_cluster_cov = np.cov(self.X.T) + np.eye(self.X.shape[1]) * 0.01

        likelihood_new = st.multivariate_normal.pdf(
            self.X[i], new_cluster_mean, new_cluster_cov
        )
        return self.__alpha * likelihood_new

    def calculate_condicional_prob(self, i):
        probs = np.zeros((self.num_clusters + 1, 1))
        keys, counts = np.unique(self.labels, return_counts=True)
        for key, n_k in list(zip(keys, counts)):
            index = np.where(self.labels == key)
            data = self.X[index]

            if n_k < 1:
                probs[key] = 1.0e-10  # Avoid division by zero
            else:
                cluster_mean = np.mean(data, 0)
                cluster_cov = np.cov(data.T)

                likelihood = st.multivariate_normal.pdf(
                    self.X[i], cluster_mean, cluster_cov
                )

                friends = np.sum(self.constraints[i, index] == 1)
                stranger = np.sum(self.constraints[i, index] != 1)

                probs[key] = (
                    likelihood
                    * n_k
                    * (self.__p**friends)
                    * ((1 - self.__p) ** stranger)
                )

        probs[-1] = self.should_create_new_cluster(i)
        probs = probs / np.sqrt(np.sum(probs**2))  # Normalize
        return probs

    def __update_centroids(self):
        """
        Calculate the new centroids

        Remove empty clusters, adjust the number of clusters and update the centroids
        """
        keys, counts = np.unique(self.labels, return_counts=True)
        len_clusters = np.zeros((self.num_clusters + 1), dtype=np.uint64)
        len_clusters[keys] = counts

        to_delete = np.where(len_clusters[: self.num_clusters] == 0)
        keys = np.delete(keys, to_delete)

        new_index = np.arange(keys.shape[0])
        for i, key in enumerate(keys):
            self.labels[self.labels == key] = new_index[i]

        self.num_clusters = keys.shape[0]
        self.__centroids = np.zeros((self.num_clusters, self.X.shape[1]))

        for i in range(self.num_clusters):
            self.__centroids[i] = np.mean(self.X[self.labels == i], 0)

    def fit(self):
        self.__initialize_centroids()

        for iteration in range(self.max_iter):
            self.__prev_centroids = self.__centroids.copy()
            condicional_prob = np.zeros((self.X.shape[0], self.num_clusters + 1))

            for i in range(self.X.shape[0]):
                condicional_prob[i] = self.calculate_condicional_prob(i)

            self.labels = np.argmax(condicional_prob, 1)
            self.__update_centroids()

            if self.stop_criteria(iteration):
                break
