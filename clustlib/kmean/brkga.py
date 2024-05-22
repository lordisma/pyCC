import numpy as np
import copy as cp
import math

from itertools import compress
from typing import Sequence
from ..utils.matrix import ConstraintMatrix

from clustlib.model import BaseEstimator


class BRKGA(BaseEstimator):
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

        self.run()

    # Funcionalidad para inicializar la poblacion de manera aleatoria
    def init_population(self):
        self._population = np.random.rand(self._population_size, self._dim)

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
            [self.fitness(solution) for solution in self._population]
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

    # Operador de cruce aleatorio
    def uniform_crossover_operator(self, parent1, parent2):
        # Obtenemos el vector de probabilidades de herdar de parent1 y resolvemos las probabilidades
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]

        # Creamos el nuevo cromosoma como una copia de parent1
        new_cromosome = cp.deepcopy(parent1)

        # Copiamos los genes de parent2 indicados por las probabilidades obtenidas
        new_cromosome[v] = parent2[v]

        return new_cromosome

    def matching_crossover_operator(self, parent1, parent2):
        # Decodificamos los padres
        decoded_p1 = self.decode_single_random_key(parent1)
        decoded_p2 = self.decode_single_random_key(parent2)

        # Obtenemos las posiciones conicidesntes y no coincidentes de ambos padres
        matches = np.where(decoded_p1 == decoded_p2)
        non_matches = np.where(decoded_p1 != decoded_p2)

        # El nuevo individuo hereda las posiciones coincidentes (calculando la media)
        new_cromosome = self.uniform_crossover_operator(parent1, parent2)
        new_cromosome[matches] = (parent1[matches] + parent2[matches]) / 2

        return new_cromosome

    def get_offspring(self, elite, non_elite, offspring_size):
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
            new_cromosome = self.uniform_crossover_operator(
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

    # Cuerpo principal del AG
    def run(self, ls=False):
        # Initialize the population
        self._evals_done = 0
        self.init_population()
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
                offspring = self.get_offspring(elite, non_elite, offspring_size)

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

        return self._best
