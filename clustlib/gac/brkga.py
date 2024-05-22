import numpy as np
import copy as cp
import math

from itertools import compress
from typing import Sequence
from ..utils.matrix import ConstraintMatrix

from clustlib.gac.base import GeneticClustering


class BRKGA(GeneticClustering):
    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
        population_size=20,
        percentage_elite=0.3,
        probability_mutation=0.2,
        pbt_inherit=None,
        mu=None,
    ):
        self._delta_centroid = None
        self.n_clusters = n_clusters
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.constraints = ConstraintMatrix(constraints)
        self._population_size = population_size
        self.solution_archive = None

        self._num_elite = int(self._population_size * percentage_elite)
        self._num_mutants = int(self._population_size * probability_mutation)

        # TODO: Extract value of ptr_elite, pbt_mutation and pbt_inherit
        # check if this need to be here
        self._pbt_inherit = pbt_inherit
        self._mu = mu
        self._evals_done = 0

    def fit(self, X, y=None, logger=None):
        self._data = X
        self._ml = self.constraints.get_ml_constraints()
        self._cl = self.constraints.get_cl_constraints()
        self._dim = X.shape[0]

        # Initialize the population
        self._evals_done = 0
        self.create_population()
        fitness = self.get_fitness()[0]
        sorted_fitness = np.argsort(fitness)
        self._population = self._population[sorted_fitness, :]
        self._best = cp.deepcopy(self.decode_single_random_key(self._population[0, :]))
        self._best_fitness = fitness[sorted_fitness[0]]

        offspring_size = self._population_size - (self._num_elite + self._num_mutants)
        if offspring_size < 0:
            offspring_size = 0

        # Run until max iteration and max evaluations are reached
        for _ in range(self.max_iter):
            # Get the best of each generation
            elite = self._population[: self._num_elite, :]
            non_elite = self._population[self._num_elite :, :]

            # Create mutants for next generation
            mutants = self.mutation()

            # Generar los descendientes de la nueva generacion cruzando los miembros de la elite
            # con el resto de individuos
            if offspring_size > 0:
                offspring = self.offspring(elite, non_elite, offspring_size)

                # Substitute the population with the new generation
                self._population[self._num_elite :, :] = np.vstack((offspring, mutants))
            else:
                self._population = np.vstack((elite, mutants))

            # Evaluate the new population
            self.calculate_fitness()
            new_best = self._population[self._population_fitness_sorted[0]]
            new_best_fitness = self._population_fitness[
                self._population_fitness_sorted[0]
            ]
            self._population = self._population[sorted_fitness, :]

            if new_best_fitness < self._best_fitness:
                self._best = cp.deepcopy(self.decode_single_random_key(new_best))
                self._best_fitness = new_best_fitness

        labels = self.get_labels()
        self.get_centroids(labels)

        return self

    def create_population(self):
        self._population = np.random.rand(self._population_size, self._dim)

    def crossover(self, parent1, parent2):
        # Get the genes that will be maintained from parent 2
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]
        new_cromosome = cp.deepcopy(parent1)
        new_cromosome[v] = parent2[v]

        return new_cromosome

    def offspring(self, elite, non_elite, offspring_size):
        # Obtenemos listas de indices aleatorios asociados a cromosomas elite y no-elite
        elite_cromosomes_index = np.random.randint(elite.shape[0], size=offspring_size)
        non_elite_cromosomes_index = np.random.randint(
            non_elite.shape[0], size=offspring_size
        )

        # Inicializamos la descendencia vacia
        offspring = np.empty((offspring_size, self._dim))

        # Generamos los nuevos inidividuos
        for i in range(offspring_size):
            # Obtenemos cada nuevo inidividuo como un cruce entre un cromosoma elitista y
            # uno no elitista
            new_cromosome = self.crossover(
                elite[elite_cromosomes_index[i], :],
                non_elite[non_elite_cromosomes_index[i], :],
            )

            # Almacenamos el nuevo individuo
            offspring[i, :] = new_cromosome

        return offspring

    def mutation(self):
        """
        Create new generation of mutants
        """
        return np.random.rand(self._num_mutants, self._dim)
