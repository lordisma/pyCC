import numpy as np
import random
import copy as cp


from ..utils.matrix import ConstraintMatrix
from typing import Sequence
from clustlib.model import BaseEstimator


class DILS(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_init_centroids=None,
        constraints: Sequence[Sequence] = None,
        probability=0.2,
        similarity_threshold=0.5,
        segment_size=10,
    ):
        self.constraints = ConstraintMatrix(constraints)
        self._result_nb_clust = n_clusters
        self._evals_done = 0
        self._pbt_inherit = probability
        self._similarity_threshold = similarity_threshold
        self._segment_size = segment_size
        self._max_iter = max_iter

    def init_ils(self):
        self._best_solution = np.random.randint(
            0, self._result_nb_clust, (2, self._dim)
        )
        self._best_fitness = np.empty(2)
        self._best_fitness[0] = self.get_single_fitness(self._best_solution[0, :])[0]
        self._best_fitness[1] = self.get_single_fitness(self._best_solution[1, :])[0]

    # TODO: This method shoulb be implemented in the base class
    # even better if we could parametrize this function so the
    # user can choose how to calculate the fitness.
    def get_single_fitness(self, cromosome):
        current_clustering = cromosome
        total_mean_distance = 0
        nb_clusters = len(set(current_clustering))

        # Para cada cluster en el clustering actual
        for j in set(current_clustering):
            # Obtener las instancias asociadas al cluster
            clust = self._data[current_clustering == j, :]

            if clust.shape[0] > 1:
                # Obtenemos la distancia media intra-cluster
                tot = 0.0
                for k in range(clust.shape[0] - 1):
                    tot += ((((clust[k + 1 :] - clust[k]) ** 2).sum(1)) ** 0.5).sum()

                avg = tot / ((clust.shape[0] - 1) * (clust.shape[0]) / 2.0)
                # Acumular la distancia media
                total_mean_distance += avg

        # Inicializamos el numero de restricciones que no se satisfacen
        infeasability = 0

        # Calculamos el numero de restricciones must-link que no se satisfacen
        for c in range(np.shape(self._ml)[0]):
            if current_clustering[self._ml[c][0]] != current_clustering[self._ml[c][1]]:
                infeasability += 1

        # Calculamos el numero de restricciones cannot-link que no se satisfacen
        for c in range(np.shape(self._cl)[0]):
            if current_clustering[self._cl[c][0]] == current_clustering[self._cl[c][1]]:
                infeasability += 1

        # Calcular el valor de la funcion fitness
        distance = total_mean_distance / nb_clusters
        penalty = distance * infeasability
        fitness = distance + penalty

        # Aumentar en uno el contador de evaluacions de la funcion objetivo
        self._evals_done += 1
        return fitness, distance, penalty

    def segment_mutation_operator(self, chromosome):
        segment_start = np.random.randint(self._dim)
        segment_end = (segment_start + self._segment_size) % self._dim
        new_segment = np.random.randint(0, self._result_nb_clust, self._segment_size)

        if segment_start < segment_end:
            chromosome[segment_start:segment_end] = new_segment
        else:
            chromosome[segment_start:] = new_segment[: self._dim - segment_start]
            chromosome[:segment_end] = new_segment[self._dim - segment_start :]
        return chromosome

    def uniform_crossover_operator(self, parent1, parent2):
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]
        new_cromosome = cp.deepcopy(parent1)
        new_cromosome[v] = parent2[v]
        return new_cromosome

    def local_search(self, chromosome):
        generated = 0
        random_index_list = np.array(range(self._dim))
        random.shuffle(random_index_list)
        ril_ind = 0
        fitness = self.get_single_fitness(chromosome)[0]

        while generated < self._max_neighbors:
            object_index = random_index_list[ril_ind]
            original_label = chromosome[object_index]
            other_labels = np.delete(
                np.array(range(self._result_nb_clust)), original_label
            )
            random.shuffle(other_labels)

            for label in other_labels:
                generated += 1
                chromosome[object_index] = label
                new_fitness = self.get_single_fitness(chromosome)[0]

                if new_fitness < fitness:
                    fitness = new_fitness
                    break
                else:
                    chromosome[object_index] = original_label

            if ril_ind == self._dim - 1:
                random.shuffle(random_index_list)
                ril_ind = 0
            else:
                ril_ind += 1

        return chromosome, fitness

    def fit(self, dataset, labels):
        self._data = dataset
        self._dim = self._data.shape[0]
        self.init_ils()

        while self._evals_done < self._max_iter:
            worst = np.argmax(self._best_fitness)
            best = (worst + 1) % 2

            new_chromosome = self.uniform_crossover_operator(
                self._best_solution[best], self._best_solution[worst]
            )
            mutant = self.segment_mutation_operator(new_chromosome)
            improved_mutant, improved_mutant_fitness = self.local_search(mutant)

            if improved_mutant_fitness < self._best_fitness[worst]:
                self._best_solution[worst] = improved_mutant
                self._best_fitness[worst] = improved_mutant_fitness

            if (
                self._best_fitness[best] - self._best_fitness[worst]
                > self._best_fitness[best] * self._similarity_threshold
            ):
                worst = np.argmax(self._best_fitness)
                self._best_solution[worst, :] = np.random.randint(
                    0, self._result_nb_clust, self._dim
                )
                self._best_fitness[worst] = self.get_single_fitness(
                    self._best_solution[worst, :]
                )[0]

        best = np.argmin(self._best_fitness)
        return self._best_solution[best, :]
