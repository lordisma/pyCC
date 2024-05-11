from typing import Sequence
import numpy as np
from clustlib.model import BaseEstimator
import pygad as pg
import math
from ..utils.matrix import ConstraintMatrix
from itertools import compress
import scipy

import logging


class ShadeCC(BaseEstimator):
    solution_archive: np.ndarray

    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
        population_size=20,
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
        self.population_size = population_size
        self.solution_archive = None

    def select_parents(self, fitness, num_parents, ga_instance):
        fitness_sorted = np.argsort(fitness)

        # Get random solution from the population
        idx_element = np.random.randint(0, self.population_size)
        x_i = fitness_sorted[idx_element]

        # Get a random "best" solution from the population
        percentange_best = np.random.uniform(1.0 / self.population_size, 0.2)
        idx_best = np.random.randint(
            0, np.ceil(self.population_size * percentange_best)
        )
        x_best = fitness_sorted[idx_best]

        r1_idx = np.random.randint(0, self.population_size)
        while r1_idx == idx_best:
            r1_idx = np.random.randint(0, self.population_size)
        x_r1 = fitness_sorted[r1_idx]

        archive_size = self.solution_archive.shape[0]
        r2_idx = np.random.randint(0, self.population_size + archive_size)
        while r2_idx == r1_idx:
            r2_idx = np.random.randint(0, self.population_size + archive_size)

        population = ga_instance.population
        if r2_idx < self.population_size:
            # Get it from the population
            x_r2 = fitness_sorted[r2_idx]

            population = ga_instance.population
            return population[[x_best, x_r1, x_r2]].copy(), np.array(
                [fitness[x_best], fitness[x_r1], fitness[x_r2]]
            )
        else:
            r2_idx -= self.population_size
            # Get it from the external archive
            if r2_idx > self.solution_archive.shape[0] - 1:
                r2_idx = np.random.randint(0, self.solution_archive.shape[0])

            archive_sol = self.solution_archive[r2_idx]
            fitness_archive = self.fitness(ga_instance, archive_sol, r2_idx)
            return np.ndarray(
                [population[x_i], population[x_best], population[x_r1], archive_sol]
            ), np.array([fitness[x_i], fitness[x_best], fitness[x_r1], fitness_archive])

    def crossover(self, parents, f_i, cr_i):
        """Crossover function for the genetic algorithm

        This function will do a single point crossover between the parents to generate the offspring,
        however, SHADE uses three parents to generate the offspring
        """
        element, best, r1, r2 = parents

        mutant = element + f_i * (best - element) + f_i * (r1 - r2)
        mutant = np.clip(mutant, 0, 1)

        cross_points_1 = np.random.rand(len(element)) <= cr_i
        cross_points_2 = np.array(range(len(element))) == np.random.randint(
            len(element), size=len(element)
        )
        cross_points = np.logical_or(cross_points_1, cross_points_2)

        mutant = np.where(cross_points, mutant, element)
        return mutant

        return np.zeros(offspring_size)

    def save_adaptive(self, delta_fitness, cr_i, f_i):
        # Create the S_CR, S_F and delta_fitness
        if self._sf is None:
            self._sf = np.empty((0, 0))

        if self._s_cr is None:
            self._s_cr = np.empty((0, 0))

        if self.fitness_delta is None:
            self.fitness_delta = np.empty((0, 0))

        self._sf = np.append(self._sf, f_i)
        self._s_cr = np.append(self._s_cr, cr_i)
        self.fitness_delta = np.append(self.fitness_delta, delta_fitness)

        pass

    def update_adaptive(self, index):
        # Get the pounderated weighted mean of the fitness
        w_k = self.fitness_delta / self.fitness_delta.sum()

        # Get the mean ponderated to update H
        mean_wa = (w_k * self._s_cr).sum()
        self._h_record_CR = np.append(self._h_record_CR, mean_wa)

        # Calculamos la media ponderada de Lehmer de S_F para actualizar H
        mean_wl = (w_k * (self._sf**2)).sum() / (w_k * self._sf).sum()
        self._h_record_F = np.append(self._h_record_CR, mean_wl)

        pass

    def create_adaptive_parameter(self):
        r_i = np.random.randint(0, self.population_size)

        while (
            f_i := scipy.stats.cauchy.rvs(
                loc=self._h_record_F[r_i],
                scale=0.1,
            )
        ) <= 0 and f_i > 1.0:
            continue

        while (
            cr_i := np.random.normal(self._h_record_CR[r_i], 0.1)
        ) <= 0 and cr_i > 1.0:
            continue

        return cr_i, f_i

    def mutation(self, current_idx):
        """Mutation function

        This function will mutate the solution using, it will select the parents and then
        it will generate the mutant solution using the crossover function
        """

        # Get the parents
        parents, fitness_parents = self.select_parents(
            self._population_fitness, 3, self.ga_instance
        )

        cr_i, f_i = self.create_adaptive_parameter()
        mutant = self.crossover(parents, f_i, cr_i)

        return mutant, cr_i, f_i

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

    def create_population(self):
        X_size = self.dataset.shape[0]
        self._external_archive = np.zeros((self.population_size, X_size))
        self._population = np.random.rand(self.population_size, X_size)
        self._h_record_CR = np.full(self.population_size, 0.5)
        self._h_record_F = np.full(self.population_size, 0.5)
        self._population_fitness = np.zeros(self.population_size)

        self._next_population = np.zeros(self.population.shape)
        self._next_population_fitness = np.zeros(self.population_size)

    def fit(self, X, y=None, logger=None):
        num_genes = X.shape[0]
        self.dataset = X
        self.solution_archive = np.zeros((0, num_genes))

        self.create_population()
        self.calculate_fitness()

        for iteration in range(self.max_iter):
            # Restart the auxiliar variables for the adaptive parameters
            self._s_cr = np.zeros((0, 0))
            self._sf = np.zeros((0, 0))
            self.fitness_delta = np.zeros((0, 0))

            for current_element in range(self.population_size):
                mutant, cr_i, f_i = self.mutation(current_element)

                mutant_fitness = self.fitness(mutant)
                current_fitness = self._population_fitness[current_element]

                if mutant_fitness < current_fitness:
                    # If the mutant is better than the current solution
                    # we need to update the adaptive parameters
                    self._next_population[current_element] = mutant
                    self._next_population_fitness[current_element] = mutant_fitness

                    self.save_adaptive(current_fitness - mutant_fitness, cr_i, f_i)
                else:
                    self._next_population[current_element] = self.population[
                        current_element
                    ]
                    self._next_population_fitness[current_element] = current_fitness

            # Substitute the population with the new one
            self._population = self._next_population
            self._population_fitness = self._next_population_fitness
            self._population_fitness_sorted = np.argsort(current_fitness)

            if len(self._s_cr) > 0 and len(self._sf) > 0:
                self.update_adaptive(iteration % self.population_size)

        labels = self.get_labels()
        self.get_centroids(labels)

        return self

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

    def local_search(self):
        raise NotImplementedError

    def search_for_movement(self):
        raise NotImplementedError
