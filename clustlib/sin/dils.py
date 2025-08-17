import random
from typing import Sequence

import numpy as np
from scipy.spatial.distance import pdist

from clustlib.model import BaseEstimator

from ..utils.distance import match_distance


class DILS(BaseEstimator):
    """Dual Intra-cluster Local Search (DILS) clustering algorithm.

    This algorithm do an iterative local search using a dual approach of bes and worst
    chromosomes. It generate a random mutation in the worst chromosome to explore

    Attributes:
        n_clusters (int): The number of clusters to form.
        init (str): Initialization method ('random' or 'custom').
        distance (callable): Distance metric function.
        tol (float): Convergence tolerance.
        custom_initial_centroids (numpy.ndarray, optional): Custom initial centroids.
        constraints (Sequence[Sequence], optional): Constraints for clustering.
        probability (float): Probability for crossover.
        similarity_threshold (float): Threshold for similarity in fitness.
        mutation_size (int): Size of the mutation segment.
        local_search_max_iter (int): Maximum iterations for local search.
        best (numpy.ndarray): Best chromosome found.
        worst (numpy.ndarray): Worst chromosome found.
        _fitness (numpy.ndarray): Fitness values of the chromosomes.

    """

    def __init__(
        self,
        n_clusters=8,
        init="random",
        distance="euclidean",
        max_iter=300,
        tol=1e-4,
        custom_init_centroids=None,
        constraints: Sequence[Sequence] | None = None,
        probability=0.2,
        similarity_threshold=0.5,
        mutation_size=10,
        local_search_max_iter=10,
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
        self.local_search_max_iter = local_search_max_iter

        self.best = None
        self.worst = None
        self._fitness = None

    def initialize(self):
        """Initialize the chromosomes and their fitness values.

        This method initializes two chromosomes with random cluster assignments and
        calculates their fitness values.
        """
        cromosomes = np.random.randint(0, self.n_clusters, (2, self.X.shape[0]))
        self._fitness = np.empty(2)
        self._fitness[0] = self.get_single_fitness(cromosomes[0, :])
        self._fitness[1] = self.get_single_fitness(cromosomes[1, :])

        self.best = cromosomes[np.argmin(self._fitness)]
        self.worst = cromosomes[np.argmax(self._fitness)]

    def _intra_cluster_distance(self, labels):
        """Calculate the intra-cluster distance.

        This method calculates the average distance between all points in the same
        cluster.

        Args:
            labels (numpy.ndarray): The labels of the clusters.

        Returns:
            float: The average intra-cluster distance.

        """
        result = 0

        if self.n_clusters == 1:
            result = pdist(self.X, metric=lambda u, v: self.distance(u - v))
            return result.mean()

        if len(labels) != self.X.shape[0]:
            raise ValueError("Labels must match the number of data points.")

        for j in np.unique(labels):
            if self.X[labels == j, :].shape[0] < 2:
                # If there is only one point in the cluster, distance is zero
                result += 0
                continue

            pdist_matrix = pdist(
                self.X[labels == j, :], metric=lambda u, v: self.distance(v - u)
            )
            result += pdist_matrix.mean()

        return result / self.n_clusters if self.n_clusters > 0 else 0.0

    def ml_infeasability(self, cromosome):
        """Must-link infeasibility.

        Calculate the infeasibility of the current clustering based on must-link
        constraints.

        Args:
            cromosome (numpy.ndarray): The current clustering labels.

        Returns:
            int: The number of must-link constraints that are not satisfied.

        """
        infeasability = 0

        for x in range(self.X.shape[0]):
            ml_constraints = np.argwhere(self.constraints[x] > 0).flatten()

            infeasability += np.sum(cromosome[ml_constraints] != cromosome[x])

        return infeasability // 2  # Each must-link constraint is counted twice

    def cl_infeasability(self, cromosome):
        """Cannot-link infeasibility.

        Calculate the infeasibility of the current clustering based on cannot-link
        constraints.

        Args:
            cromosome (numpy.ndarray): The current clustering labels.

        Returns:
            int: The number of cannot-link constraints that are not satisfied.

        """
        infeasability = 0

        for x in range(self.X.shape[0]):
            cl_constraints = np.argwhere(self.constraints[x] < 0).flatten()

            infeasability += np.sum(cromosome[cl_constraints] == cromosome[x])

        return infeasability // 2

    def get_single_fitness(self, cromosome):
        """Calculate the fitness of a single chromosome.

        Args:
            cromosome (numpy.ndarray): The chromosome to evaluate.

        Returns:
            float: The fitness value of the chromosome.

        """
        distance = self._intra_cluster_distance(cromosome)
        ml_infeasability = self.ml_infeasability(cromosome)
        cl_infeasability = self.cl_infeasability(cromosome)

        penalty = distance * (ml_infeasability + cl_infeasability)
        fitness = distance + penalty

        return fitness

    def mutation(self, chromosome):
        """Perform mutation on a chromosome.

        Args:
            chromosome (numpy.ndarray): The chromosome to mutate.

        Returns:
            numpy.ndarray: The mutated chromosome.

        """
        n = self.X.shape[0]
        segment_start = np.random.randint(n)
        segment_end = (segment_start + self._mutation_size) % n
        new_segment = np.random.randint(0, self.n_clusters, self._mutation_size)

        if segment_start < segment_end:
            chromosome[segment_start:segment_end] = new_segment
        else:
            chromosome[segment_start:] = new_segment[: n - segment_start]
            chromosome[:segment_end] = new_segment[n - segment_start :]
        return chromosome

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents.

        Args:
            parent1 (numpy.ndarray): The first parent chromosome.
            parent2 (numpy.ndarray): The second parent chromosome.

        Returns:
            numpy.ndarray: The new chromosome created by crossover.

        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parent chromosomes must have the same shape.")

        v = np.argwhere(np.random.rand(self.X.shape[0]) > self._probability)
        new_cromosome = np.copy(parent1)
        new_cromosome[v] = parent2[v]
        return new_cromosome

    def local_search(self, chromosome):
        """Perform local search on a chromosome.

        Args:
            chromosome (numpy.ndarray): The chromosome to improve.

        Returns:
            numpy.ndarray: The improved chromosome.

        """
        index_list = np.arange(len(chromosome))
        fitness = self.get_single_fitness(chromosome)
        iterations = 0

        random.shuffle(index_list)

        for index in index_list[: self.local_search_max_iter]:
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

            if iterations == self.local_search_max_iter:
                break

        return chromosome

    def _update(self):
        new_chromosome = self.crossover(self.best, self.worst)
        mutant = self.mutation(new_chromosome)
        improved_mutant = self.local_search(mutant)
        improved_mutant_fitness = self.get_single_fitness(improved_mutant)

        if improved_mutant_fitness < np.max(self._fitness):
            if improved_mutant_fitness < np.min(self._fitness):
                self.worst = self.best
                self.best = improved_mutant
            else:
                self.worst = improved_mutant
            self._fitness[np.argmax(self._fitness)] = improved_mutant_fitness

        threshold = np.min(self._fitness) * self._threshold
        if (np.max(self._fitness) - np.min(self._fitness)) > threshold:
            worst = np.argmax(self._fitness)
            self.worst = np.random.randint(0, self.n_clusters, self.X.shape[0])
            self._fitness[worst] = self.get_single_fitness(self.worst)

        self._labels = self.best
        self.calculate_centroids()

    def calculate_centroids(self):
        """Calculate the centroids of the clusters based on the current labels.

        Returns:
            numpy.ndarray: The centroids of the clusters.

        """
        centroids = np.zeros((self.n_clusters, self.X.shape[1]))

        for i in range(self.n_clusters):
            centroids[i] = np.mean(self.X[self._labels == i, :], axis=0)

        self.centroids = centroids

    def _fit(self):
        """Fit the model to the data.

        Returns:
            numpy.ndarray: The best chromosome found.

        """
        self.initialize()
        iteration = 0

        while not self.stop_criteria(iteration):
            self.update()
            iteration += 1

        return self._labels
