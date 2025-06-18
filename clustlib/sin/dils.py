import numpy as np
import random
from scipy.spatial.distance import pdist

from ..utils.distance import match_distance
from typing import Sequence
from clustlib.model import BaseEstimator


class DILS(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        init = "random",
        distance = "euclidean",
        max_iter=300,
        tol = 1e-4,
        custom_init_centroids = None,
        constraints: Sequence[Sequence] = None,
        probability=0.2,
        similarity_threshold=0.5,
        mutation_size=10,
    ):
        self.init = init
        self.distance = match_distance(distance)
        self.tol = tol
        self.custom_initial_centroids = custom_init_centroids
        self.constraints = constraints

        self.n_clusters = n_clusters
        self._evals_done = 0
        self._probability = probability
        self._threshold = similarity_threshold
        self._mutation_size = mutation_size
        self.max_iter = max_iter

    def initialize(self):
        """Initialize the chromosomes and their fitness values.

        This method initializes two chromosomes with random cluster assignments and calculates their fitness values.
        """
        cromosomes = np.random.randint(
            0, self.n_clusters, (2, self.X.shape[0])
        )
        self._fitness = np.empty(2)
        self._fitness[0] = self.get_single_fitness(cromosomes[0, :])
        self._fitness[1] = self.get_single_fitness(cromosomes[1, :])

        self.best = cromosomes[np.argmin(self._fitness)]
        self.worst = cromosomes[np.argmax(self._fitness)]

    def _intra_cluster_distance(self, labels):
        """Calculate the intra-cluster distance.

        This method calculates the average distance between all points in the same cluster.
        Parameters
        __________
        labels: numpy.ndarray
            The labels of the clusters.
        Returns
        _______
        result: float
            The average intra-cluster distance.
        
        """
        result = 0

        if self.n_clusters == 1:
            return pdist(self.X, metric=self.distance).mean()

        for j in labels.unique():
            if np.any(labels == j):
                result += pdist(self.X[labels == j, :], metric=self.distance).mean()

        return result / self.n_clusters if self.n_clusters > 0 else 0.0
    
    def _ml_infeasability(self, cromosome):
        """Calculate the infeasibility of the current clustering based on must-link constraints.

        Parameters
        __________
        current_clustering: numpy.ndarray
            The current clustering labels.

        Returns
        _______
        infeasability: int
            The number of must-link constraints that are not satisfied.
        """
        infeasability = 0

        for x in range(self.X.shape[0]):
            ml_constraints = np.argwhere(self.constraints[x] > 0).flatten()

            infeasability += np.sum(cromosome[ml_constraints] != cromosome[x])

        return infeasability // 2  # Each must-link constraint is counted twice, once for each element in the pair.
    
    def _cl_infeasability(self, cromosome):
        """Calculate the infeasibility of the current clustering based on cannot-link constraints.

        Parameters
        __________
        current_clustering: numpy.ndarray
            The current clustering labels.

        Returns
        _______
        infeasability: int
            The number of cannot-link constraints that are not satisfied.
        """
        infeasability = 0

        for x in range(self.X.shape[0]):
            cl_constraints = np.argwhere(self.constraints[x] < 0).flatten()

            infeasability += np.sum(cromosome[cl_constraints] != cromosome[x])

        return infeasability // 2
        

    def get_single_fitness(self, cromosome):
        """Calculate the fitness of a single chromosome.

        Parameters
        __________
        cromosome: numpy.ndarray
            The chromosome to evaluate.
        Returns
        _______
        fitness: float
            The fitness value of the chromosome.
        """
        distance = self._intra_cluster_distance(cromosome)
        ml_infeasability = self._ml_infeasability(cromosome)
        cl_infeasability = self._cl_infeasability(cromosome)

        penalty = distance * (ml_infeasability + cl_infeasability)
        fitness = distance + penalty

        return fitness

    def mutation(self, chromosome):
        n = self.X.shape[0]
        segment_start = np.random.randint(n)
        segment_end = (segment_start + self._segment_size) % n
        new_segment = np.random.randint(0, self.n_clusters, self._segment_size)

        if segment_start < segment_end:
            chromosome[segment_start:segment_end] = new_segment
        else:
            chromosome[segment_start:] = new_segment[: n - segment_start]
            chromosome[:segment_end] = new_segment[n - segment_start :]
        return chromosome

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents.

        Parameters
        __________
        parent1: numpy.ndarray
            The first parent chromosome.
        parent2: numpy.ndarray
            The second parent chromosome.

        Returns
        _______
        new_cromosome: numpy.ndarray
            The new chromosome created by crossover.
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parent chromosomes must have the same shape.")

        v = np.argwhere(np.random.rand(self.X.shape[0]) > self._probability)
        new_cromosome = np.copy(parent1)
        new_cromosome[v] = parent2[v]
        return new_cromosome

    def local_search(self, chromosome, max_iter):
        index_list = np.arange(len(chromosome))
        fitness = self.get_single_fitness(chromosome)
        iterations = 0

        random.shuffle(index_list)

        for index in index_list[:max_iter]:
            original_label = chromosome[index]
            labels = np.arange(self.n_clusters)

            for label in labels:
                if label == original_label:
                    continue

                iterations += 1

                chromosome[index] = label
                new_fitness = self.get_single_fitness(chromosome)

                if new_fitness < fitness:
                    fitness = new_fitness
                    break
                else:
                    chromosome[index] = original_label

            if iterations == max_iter:
                break

        return chromosome

    def _fit(self):
        self.initialize()
        iteration = 0

        while not self.stop_criteria(iteration):
            new_chromosome = self.crossover(
                self.best, self.worst
            )

            mutant = self.mutation(new_chromosome)
            improved_mutant = self.local_search(mutant)
            improved_mutant_fitness = self.get_single_fitness(improved_mutant)

            if improved_mutant_fitness < np.max(self._fitness):
                self.worst = improved_mutant
                self._fitness[np.argmax(self._fitness)] = improved_mutant_fitness

            threshold = np.min(self._fitness) * self._threshold

            if (np.max(self._fitness) - np.min(self._fitness)) > threshold:
                worst = np.argmax(self._fitness)
                self.worst = np.random.randint(0, self.n_clusters, self.X.shape[0])
                self._fitness[worst] = self.get_single_fitness(self.worst)
            iteration += 1

        return self.best
