import logging
import math
from typing import Sequence

import numpy as np

from clustlib.gac._base import GeneticClustering

logger = logging.getLogger(__name__)


class BRKGA(GeneticClustering):
    """BRKGA (Biased Random-Key Genetic Algorithm).

    BRKGA is a genetic algorithm adapted for clustering with constraints.
    It uses a random-key encoding to represent solutions and biased genetic operators
    to generate new populations that optimize data partitioning.

    This algorithm inherits from the base class GeneticClustering.

    Attributes:
        n_clusters (int): Number of target clusters.
        init (str): Method for initializing centroids.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for the convergence criterion.
        constraints (Sequence[Sequence]): Must-link and cannot-link constraints.
        population_size (int): Size of the genetic population.
        percentage_elite (float): Percentage of individuals considered elite.
        probability_mutation (float): Percentage of mutants in each generation.
        pbt_inherit (float): Probability of inheritance in the crossover operator.

    """

    def __init__(
        self,
        constraints: Sequence[Sequence],
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        population_size=20,
        percentage_elite=0.3,
        probability_mutation=0.2,
        pbt_inherit=0.1,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.constraints = constraints
        self._population_size = population_size
        self._dim = constraints.shape[0]

        self._num_elite = math.ceil(self._population_size * percentage_elite)
        self._num_mutants = math.ceil(self._population_size * probability_mutation)

        self._pbt_inherit = pbt_inherit

    def _update(self):
        mutants = self.mutation()

        normal_population = (
            self.population.shape[0] - self._num_elite - self._num_mutants
        )
        normal_population = max(normal_population, 0)
        if normal_population > 0:
            offspring = self.offspring(normal_population)
            self.population[self._num_elite :, :] = np.vstack((offspring, mutants))
        else:
            self.population[self._num_elite :, :] = mutants

        self.calculate_fitness()
        self._labels = self.decode_solution(self.population[0, :])
        self.centroids = self.get_centroids(self._labels)

    def _convergence(self):
        if self._delta is None:
            logger.debug("Delta is None, convergence cannot be checked.")
            return False

        return np.linalg.norm(self._delta) < self.tol

    def _fit(self):
        """Fits the BRKGA model to the given data.

        Args:
            X (ndarray): Input data matrix.
            y (ndarray, optional): True labels if available (not used).
            logger (Logger, optional): Logging object for process tracking.

        Returns:
            self: Fitted object.

        """
        self.create_population()

        iteration = 0
        while not self.stop_criteria(iteration):
            self.update()
            iteration += 1

        return self

    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos padres para generar un nuevo cromosoma.

        Args:
            parent1 (ndarray): Cromosoma del padre elitista.
            parent2 (ndarray): Cromosoma del padre no elitista.

        Returns:
            ndarray: Nuevo cromosoma generado.

        """
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]
        new_cromosome = parent1
        new_cromosome[v] = parent2[v]
        return new_cromosome

    def offspring(self, offspring_size):
        """Genera descendencia cruzando padres elite con no-elite.

        Args:
            offspring_size (int): Número de descendientes a generar.

        Returns:
            ndarray: Matriz con la descendencia generada.

        """
        elite_idx = np.random.randint(self._num_elite, size=offspring_size)
        non_elite_idx = np.random.randint(
            low=self._num_elite, high=self._population_size, size=offspring_size
        )
        offspring = np.empty((offspring_size, self._dim))

        elites = self.population[elite_idx]
        non_elites = self.population[non_elite_idx]

        i = 0
        for elite, non_elite in zip(elites, non_elites):
            offspring[i, :] = self.crossover(elite, non_elite)
            i += 1

        return offspring

    def mutation(self):
        """Crea una nueva generación de individuos mutantes.

        Returns:
            ndarray: Matriz de cromosomas mutantes.

        """
        return np.random.rand(self._num_mutants, self._dim)
