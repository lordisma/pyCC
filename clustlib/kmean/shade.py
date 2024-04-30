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
        idx_best = np.random.randint(0, np.ceil(self.population_size * percentange_best))
        x_best = fitness_sorted[idx_best]

        r1_idx = np.random.randint(0, self.population_size)
        while r1_idx == idx_best: r1_idx = np.random.randint(0, self.population_size)
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
            return population[[x_best, x_r1, x_r2]].copy(), np.array([fitness[x_best], fitness[x_r1], fitness[x_r2]])
        else:
            r2_idx -= self.population_size
            # Get it from the external archive
            if r2_idx > self.solution_archive.shape[0] - 1:
                r2_idx = np.random.randint(0, self.solution_archive.shape[0])

            archive_sol = self.solution_archive[r2_idx]
            fitness_archive = self.fitness(ga_instance, archive_sol, r2_idx)
            return np.ndarray([
                    population[x_i],
                    population[x_best], 
                    population[x_r1], 
                    archive_sol]), np.array([
                    fitness[x_i],
                    fitness[x_best], 
                    fitness[x_r1], 
                    fitness_archive])
        
    def crossover(self, parents, offspring_size, ga_instance):
        """Crossover function for the genetic algorithm

        This function will do a single point crossover between the parents to generate the offspring,
        however, SHADE uses three parents to generate the offspring
        """
        print(f"parents: {parents}, offspring_size: {offspring_size}")

        element, best, r1, r2 = parents
        r2_fitness = self.fitness(ga_instance, r2, -1)
        while (f_i := scipy.stats.cauchy.rvs(loc=r2_fitness, scale=0.1, )) <= 0 and f_i > 1.0:
            continue

        mutant = element + f_i * (best - element) + f_i * (r1 - r2)
        mutant = np.clip(mutant, 0, 1)

        # We need to calculate the H parameter


        return np.zeros(offspring_size)
    
    def save_adaptive(self, delta_fitness, cr_i, f_i):
        # Create the S_CR, S_F and delta_fitness
        if self.sf is None:
            self.sf = np.empty((0,0))

        if self.s_cr is None:
            self.s_cr = np.empty((0,0))

        if self.fitness_delta is None:
            self.fitness_delta = np.empty((0,0))
        
        self.sf = np.append(self.sf, f_i)
        self.s_cr = np.append(self.scr, cr_i)
        self.fitness_delta = np.append(self.fitness_delta, delta_fitness)
        
        pass

    def update_adaptive(self):
        # Get the pounderated weighted mean of the fitness
        w_k = self.fitness_delta / self.fitness_delta.sum()

        # Get the mean ponderated to update H
        mean_wa = (w_k * self.s_cr).sum()
        self._h_record_CR = np.append(self._h_record_CR, mean_wa)

        #Calculamos la media ponderada de Lehmer de S_F para actualizar H
        mean_wl = (w_k * (self.sf**2)).sum() / (w_k * self.sf).sum()
        self._h_record_CR = np.append(self._h_record_CR, mean_wl)

        h_index = (h_index + 1) % self._population_size
        pass


    def mutation(self, offspring_crossover, ga_instance):
        print(f"offspring_crossover: {offspring_crossover}")
        return offspring_crossover


    def decode_solution(self, solution):
        decoded = np.ceil(solution * self.n_clusters)
        decoded[decoded == 0] = 1

        return decoded - 1

    def fitness(self, ga_instance, solution, solution_idx):
        labels = self.decode_solution(solution)
        total_distance = self.distance_to_cluster(labels)
        infeasability = self.infeseability(labels)

        # FIXME: This must be a maximization problem
        fitness = total_distance + infeasability
        if math.isnan(fitness):
            raise ValueError("Fitness is NaN")
        return fitness
    
    def distance_to_cluster(self, labels):
        total_distance = 0.0

        for label in set(labels):
            data_from_cluster = self.X[labels == label, :]

            # IF there is no data in the cluster we could skip it
            # and aggregate to the distance a penalty in order to
            # avoid empty clusters. 
            # We do the same for clusters with only one data point
            # so we force the algorithm to place the data in clusters
            # more evently distributed
            if data_from_cluster.shape[0] <= 1:
                total_distance += 10.0
                continue

            distances = np.linalg.norm(data_from_cluster[1:] - data_from_cluster[:-1], axis=1)
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

            infeasability += (must_link_infeasability + cannot_link_infeasability)

        return infeasability
    

    def fit(self, X, y = None, logger = None):
        num_genes = X.shape[0]
        self.X = X
        self.solution_archive = np.zeros((0, num_genes))

        for i in range(self.max_iter):
            population_fitness = np.zeros(self.population_size)

            for j in range(self.population_size):
                solution = np.random.rand(X.shape)
                
                fitness = self.fitness(solution)
                self.solution_archive = np.append(self.solution_archive, solution)
                self.fitness_archive = np.append(self.fitness_archive, fitness)

        # ga = pg.GA(num_generations=self.max_iter, 
        #                      num_parents_mating=4, 
        #                      sol_per_pop = self.population_size, 
        #                      num_genes = num_genes, 
        #                      fitness_func = self.fitness,
        #                     #  on_parents = on_parents,
        #                     #  on_crossover=on_crossover,
        #                     #  on_mutation=on_mutation,
        #                     #  on_generation=on_generation,
        #                     #  on_fitness=on_fitness,
        #                      init_range_low=0,
        #                      init_range_high=1,
        #                      parent_selection_type = self.select_parents,
        #                      save_best_solutions=True,
        #                      save_solutions=True,
        #                      crossover_type=self.crossover,
        #                      keep_parents=3,
        #                      logger = logger)
        ga.run()
        
        solution, fitness, solution_idx = ga.best_solution()

        print(f"Solution: {solution}, fitness: {fitness}, solution_idx: {solution_idx}")
        self.final_labels = self.decode_solution(solution)

        print(f"Final labels: {self.final_labels}")
        self.centroids = self.get_centroids(self.final_labels)

        return self


    def get_centroids(self, labels):
        centroids = []

        for label in set(labels):
            data_from_cluster = self.X[labels == label, :]

            if data_from_cluster.shape[0] == 0:
                continue

            centroids.append(np.mean(data_from_cluster, axis=0))

        return np.array(centroids)

    def local_search(self):
        raise NotImplementedError

    def search_for_movement(self):
        raise NotImplementedError
