import numpy as np
import copy as cp

from typing import Sequence
from ..utils.matrix import ConstraintMatrix

from clustlib.gac.base import GeneticClustering


class BRKGA(GeneticClustering):
    """
    BRKGA (Biased Random-Key Genetic Algorithm) es un algoritmo genético adaptado al clustering con restricciones.
    Utiliza una codificación basada en claves aleatorias (random keys) para representar soluciones y operadores genéticos
    sesgados para generar nuevas poblaciones que optimizan la partición de datos.

    Este algoritmo hereda de la clase base GeneticClustering.

    Atributos:
        n_clusters (int): Número de clusters objetivo.
        init (str): Método de inicialización de centroides.
        max_iter (int): Máximo número de iteraciones.
        tol (float): Tolerancia para criterio de convergencia.
        custom_initial_centroids (array-like): Centroides definidos por el usuario (opcional).
        constraints (Sequence[Sequence]): Restricciones must-link y cannot-link.
        population_size (int): Tamaño de la población genética.
        percentage_elite (float): Porcentaje de individuos considerados elite.
        probability_mutation (float): Porcentaje de mutantes en cada generación.
        pbt_inherit (float): Probabilidad de herencia en el operador de cruce.
        mu (float): Parámetro de mutación.
    """
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

        self._pbt_inherit = pbt_inherit
        self._mu = mu
        self._evals_done = 0

    def fit(self, X, y=None, logger=None):
        """
        Ajusta el modelo BRKGA a los datos dados.

        Args:
            X (ndarray): Matriz de datos de entrada.
            y (ndarray, optional): Etiquetas reales si están disponibles (no utilizado).
            logger (Logger, optional): Objeto de logging para seguimiento del proceso.

        Returns:
            self: Objeto ajustado.
        """
        self._data = X
        self._ml = self.constraints.get_ml_constraints()
        self._cl = self.constraints.get_cl_constraints()
        self._dim = X.shape[0]

        self._evals_done = 0
        self.create_population()
        fitness = self.get_fitness()[0]
        sorted_fitness = np.argsort(fitness)
        self._population = self._population[sorted_fitness, :]
        self._best = cp.deepcopy(self.decode_single_random_key(self._population[0, :]))
        self._best_fitness = fitness[sorted_fitness[0]]

        offspring_size = self._population_size - (self._num_elite + self._num_mutants)
        offspring_size = max(offspring_size, 0)

        for _ in range(self.max_iter):
            elite = self._population[: self._num_elite, :]
            non_elite = self._population[self._num_elite :, :]

            mutants = self.mutation()

            if offspring_size > 0:
                offspring = self.offspring(elite, non_elite, offspring_size)
                self._population[self._num_elite :, :] = np.vstack((offspring, mutants))
            else:
                self._population = np.vstack((elite, mutants))

            self.calculate_fitness()
            new_best = self._population[self._population_fitness_sorted[0]]
            new_best_fitness = self._population_fitness[self._population_fitness_sorted[0]]
            self._population = self._population[sorted_fitness, :]

            if new_best_fitness < self._best_fitness:
                self._best = cp.deepcopy(self.decode_single_random_key(new_best))
                self._best_fitness = new_best_fitness

        labels = self.get_labels()
        self.get_centroids(labels)
        return self

    def create_population(self):
        """
        Inicializa la población con valores aleatorios en el rango [0, 1].
        """
        self._population = np.random.rand(self._population_size, self._dim)

    def crossover(self, parent1, parent2):
        """
        Realiza cruce entre dos padres para generar un nuevo cromosoma.

        Args:
            parent1 (ndarray): Cromosoma del padre elitista.
            parent2 (ndarray): Cromosoma del padre no elitista.

        Returns:
            ndarray: Nuevo cromosoma generado.
        """
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]
        new_cromosome = cp.deepcopy(parent1)
        new_cromosome[v] = parent2[v]
        return new_cromosome

    def offspring(self, elite, non_elite, offspring_size):
        """
        Genera descendencia cruzando padres elite con no-elite.

        Args:
            elite (ndarray): Subpoblación elitista.
            non_elite (ndarray): Subpoblación no elitista.
            offspring_size (int): Número de descendientes a generar.

        Returns:
            ndarray: Matriz con la descendencia generada.
        """
        elite_idx = np.random.randint(elite.shape[0], size=offspring_size)
        non_elite_idx = np.random.randint(non_elite.shape[0], size=offspring_size)
        offspring = np.empty((offspring_size, self._dim))

        for i in range(offspring_size):
            offspring[i, :] = self.crossover(
                elite[elite_idx[i], :], non_elite[non_elite_idx[i], :]
            )

        return offspring

    def mutation(self):
        """
        Crea una nueva generación de individuos mutantes.

        Returns:
            ndarray: Matriz de cromosomas mutantes.
        """
        return np.random.rand(self._num_mutants, self._dim)
