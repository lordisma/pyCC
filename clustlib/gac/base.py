import math

import numpy as np
from scipy.spatial.distance import pdist

from clustlib.model import BaseEstimator


class GeneticClustering(BaseEstimator):
    """GeneticClustering.

    Create the base class for the genetic clustering algorithms. This class will abstract common methods to
    all genetic algorithms like fitness evaluation, selection, crossover, infeasibility calculation, etc.
    """

    def decode_solution(self, solution):
        decoded = np.ceil(solution * self.n_clusters)
        decoded[decoded == 0] = 1

        return decoded - 1

    def fitness(self, solution):
        labels = self.decode_solution(solution)
        total_distance = self.distance_to_cluster(labels)
        infeasability = self.infeseability(labels)

        fitness = total_distance + infeasability
        if math.isnan(fitness):
            raise ValueError("Fitness is NaN")
        return fitness

    def calculate_fitness(self):
        self._population_fitness = np.array(
            [self.fitness(solution) for solution in self.population]
        )
        self.population = self.population[np.argsort(self._population_fitness), :]
        self._population_fitness = np.sort(self._population_fitness)

    def distance_to_cluster(self, labels):
        result = 0.0

        for j in np.unique(labels):
            if self.X[labels == j, :].shape[0] < 2:
                result += 10.0
                continue

            pdist_matrix = pdist(self.X[labels == j, :], metric="euclidean")
            result += pdist_matrix.mean()

        return result / self.n_clusters if self.n_clusters > 0 else np.Infinity

    def infeseability(self, labels):
        infeasability = 0

        for idx, cluster in enumerate(labels):
            ml = np.argwhere(self.constraints[idx] > 0).flatten()
            linked = np.argwhere(labels == cluster).flatten()

            ml_infeasability = ml.shape[0] - np.sum(np.isin(linked, ml))

            cl = np.argwhere(self.constraints[idx] < 0).flatten().sort()
            cl_infeasability = np.sum(np.isin(linked, cl))

            infeasability += ml_infeasability + cl_infeasability

        return infeasability

    def get_labels(self):
        best = self.population[0]
        return self.decode_solution(best)

    def get_centroids(self, labels):
        centroids = []

        for label in set(labels):
            data_from_cluster = self.X[labels == label, :]

            if data_from_cluster.shape[0] == 0:
                continue

            centroids.append(np.mean(data_from_cluster, axis=0))

        return np.array(centroids)

    def create_population(self):
        """Create the initial population for the genetic algorithm."""
        self.population = np.random.rand(self._population_size, self._dim)
        self.calculate_fitness()
        self._labels = self.decode_solution(self.population[0, :])
        self.centroids = self.get_centroids(self._labels)
