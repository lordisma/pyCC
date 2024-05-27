from clustlib.model import BaseEstimator
import math

import numpy as np
from itertools import compress


class GeneticClustering(BaseEstimator):
    """GeneticClustering

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

        # FIXME: This must be a maximization problem
        fitness = total_distance + infeasability
        if math.isnan(fitness):
            raise ValueError("Fitness is NaN")
        return fitness

    def calculate_fitness(self):
        current_fitness = np.array(
            [self.fitness(solution) for solution in self.population]
        )

        self._population_fitness = current_fitness
        self._population_fitness_sorted = np.argsort(current_fitness)

    def distance_to_cluster(self, labels):
        total_distance = 0.0

        for label in set(labels):
            data_from_cluster = self.dataset[labels == label, :]

            # IF there is no data in the cluster we could skip it
            # and aggregate to the distance a penalty in order to
            # avoid empty clusters.
            # We do the same for clusters with only one data point
            # so we force the algorithm to place the data in clusters
            # more evently distributed
            if data_from_cluster.shape[0] <= 1:
                total_distance += 10.0
                continue

            distances = np.linalg.norm(
                data_from_cluster[1:] - data_from_cluster[:-1], axis=1
            )
            average_distance = np.mean(distances)

            total_distance += average_distance

        return total_distance

    def infeseability(self, labels):
        infeasability = 0

        for idx, cluster in enumerate(labels):
            ml = set(self.constraints.get_ml_constraints(idx))
            linked = set(compress(range(len(labels)), labels == cluster))

            must_link_infeasability = 1 if len(ml - linked) > 0 else 0

            cl = set(self.constraints.get_cl_constraints(idx))
            cannot_link_infeasability = 1 if len(cl & linked) > 0 else 0

            infeasability += must_link_infeasability + cannot_link_infeasability

        return infeasability

    def get_labels(self):
        best = self._population[self._population_fitness_sorted[0]]
        return self.decode_solution(best)

    def get_centroids(self, labels):
        centroids = []

        for label in set(labels):
            data_from_cluster = self.dataset[labels == label, :]

            if data_from_cluster.shape[0] == 0:
                continue

            centroids.append(np.mean(data_from_cluster, axis=0))

        return np.array(centroids)
