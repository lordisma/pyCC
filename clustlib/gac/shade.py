import logging
from typing import Sequence

import numpy as np
import scipy

from clustlib.gac.base import GeneticClustering

logger = logging.getLogger(__name__)

class ShadeCC(GeneticClustering):
    """SHADE Clustering with Constraints (ShadeCC).

    Algoritmo genético adaptativo basado en SHADE para resolver problemas de clustering con restricciones.
    Utiliza historia de éxito para ajustar dinámicamente los parámetros de evolución diferencial.

    Attributes:
        population_size (int): Número de soluciones en la población genética.
        n_clusters (int): Número de clusters objetivo.
        init (str): Método de inicialización de centroides.
        max_iter (int): Número máximo de generaciones.
        tol (float): Tolerancia de convergencia.
        constraints (ConstraintMatrix): Restricciones must-link y cannot-link.
        solution_archive (np.ndarray): Archivo externo de soluciones para mantener diversidad.
    """

    solution_archive: np.ndarray
    percentage_best: float = 0.2

    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
        population_size=20,
        memory_size=20,
    ):
        """Inicializa el algoritmo SHADE para clustering con restricciones.

        Args:
            n_clusters (int): Número de clusters a generar.
            init (str): Método de inicialización de centroides.
            max_iter (int): Número máximo de iteraciones.
            tol (float): Tolerancia para la convergencia.
            custom_initial_centroids (Optional[np.ndarray]): Centroides definidos por el usuario.
            constraints (Sequence[Sequence]): Lista de restricciones ML y CL.
            population_size (int): Tamaño de la población genética.
        """
        self._delta_centroid = None
        self.n_clusters = n_clusters
        self._centroids_distance = np.zeros((self.n_clusters, self.n_clusters))
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.custom_initial_centroids = custom_initial_centroids
        self.centroids = None
        self.constraints = constraints
        self._population_size = population_size
        self.memory_size = memory_size
        self._num_elite = np.ceil(self._population_size * self.percentage_best).astype(int)
        self.solution_archive = None
        self._dim = self.constraints.shape[0]

    @staticmethod
    def randint(low, high=None, size=None, exclude=None):
        """Genera un número entero aleatorio, excluyendo ciertos valores.
        
        Args:
            low (int): Límite inferior del rango.
            high (int, optional): Límite superior del rango.
            size (int, optional): Número de valores a generar.
            exclude (Sequence[int], optional): Valores a excluir del rango.

        Returns:
            np.ndarray: Array de enteros aleatorios excluyendo los valores especificados.
        """
        if exclude is None:
            return np.random.randint(low, high, size)
        else:
            return np.random.choice(
                [i for i in range(low, high) if i not in exclude], size=size
            )
    
    def _convergence(self):
        if self._delta is None:
            logger.debug("Delta is None, convergence cannot be checked.")
            return False
        
        return np.abs(np.linalg.norm(self._delta)) < self.tol

    def _fit(self):
        self.create_population()
        self.create_archive()
        
        iteration = 0
        while not self.stop_criteria(iteration):
            self.update()
            iteration += 1

        return self
    
    def create_population(self):
        super().create_population()

        self._next_population = np.zeros(self.population.shape)
        self._next_population_fitness = np.zeros(self._population_size)

    def create_archive(self):
        """Inicializa el archivo externo de soluciones."""
        if self.solution_archive is None:
            self.solution_archive = np.zeros((0, self._dim))
        else:
            self.solution_archive = np.vstack((self.solution_archive, np.zeros((0, self._dim))))

        self._memory_CR = np.full(self.memory_size, 0.5)
        self._memory_F = np.full(self.memory_size, 0.5)

    def get_instances(self, parents_idx):
        """Obtiene las instancias de la población y el archivo externo.
        
        Args:
            parents_idx (np.ndarray): Índices de los padres seleccionados.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Instancias de la población y del archivo externo.
        """
        (idx, best, r1, r2) = parents_idx

        return (
            self.population[idx],
            self.population[best],
            self.population[r1],
            self.population[r2] if r2 < self._population_size else self.solution_archive[r2 - self._population_size]
        )

    def select_parents(self):
        """Selecciona padres desde la población y el archivo externo.

        Returns:
            np.ndarray: Index to the parents selected.
        """
        idx = self.randint(0, self._population_size)
        idx_best = self.randint(0, self._num_elite, exclude=[idx])
        idx_r1 = self.randint(0, self._population_size, exclude=[idx_best, idx])

        archive_size = self.solution_archive.shape[0]
        idx_r2 = self.randint(0, self._population_size + archive_size, exclude=[idx_best, idx_r1, idx])

        return np.array([idx, idx_best, idx_r1, idx_r2])

    def crossover(self, parents, f_i, cr_i):
        """Realiza cruce diferencial entre múltiples padres.

        Args:
            parents (np.ndarray): Soluciones padre [elemento, best, r1, r2].
            f_i (float): Factor de escala.
            cr_i (float): Tasa de cruce.

        Returns:
            np.ndarray: Cromosoma mutante generado.
        """
        element, best, r1, r2 = self.get_instances(parents)
        mutant = element + f_i * (best - element) + f_i * (r1 - r2)
        mutant = np.clip(mutant, 0, 1)

        cross_points_1 = np.random.rand(self._dim) <= cr_i
        cross_points_2 = np.random.rand(self._dim) < (1 / self._dim)
        cross_points = np.logical_or(cross_points_1, cross_points_2)

        return np.where(cross_points, mutant, element)

    def save_adaptive(self, delta_fitness, cr_i, f_i):
        """Guarda los parámetros exitosos para adaptación futura.

        Args:
            delta_fitness (float): Mejora obtenida en fitness.
            cr_i (float): Tasa de cruce usada.
            f_i (float): Factor de escala usado.
        """
        self._sf = np.append(self._sf, f_i)
        self._s_cr = np.append(self._s_cr, cr_i)
        self.fitness_delta = np.append(self.fitness_delta, delta_fitness)

    def update_adaptive(self):
        """Actualiza los historiales H_CR y H_F con medias ponderadas."""
        k = np.random.randint(0, self.memory_size)
        mean_wa = np.average(self.fitness_delta, weights=self._s_cr)
        self._memory_CR[k] = np.clip(mean_wa, 0.0, 1.0)

        w_k = self.fitness_delta / np.sum(self.fitness_delta)
        mean_wl = (w_k * (self._sf**2)).sum() / (w_k * self._sf).sum()
        self._memory_F[k] = np.clip(mean_wl, a_min=0.0, a_max=1.0)

    def create_adaptive_parameter(self):
        """Genera nuevos parámetros CR y F adaptativos.

        Returns:
            Tuple[float, float]: Par de (CR, F) adaptados.
        """
        r_i = np.random.randint(0, self.memory_size)
        while (
            f_i := scipy.stats.cauchy.rvs(
                loc=self._memory_F[r_i], scale=0.1
            )
        ) <= 0 and f_i > 1.0:
            continue

        while (
            cr_i := np.random.normal(self._memory_CR[r_i], 0.1)
        ) <= 0 and cr_i > 1.0:
            continue

        return cr_i, f_i

    def mutation(self):
        """Genera mutación sobre el individuo actual.

        Returns:
            Tuple[np.ndarray, float, float]: Mutante, cr_i y f_i generados.
        """
        parents = self.select_parents()
        cr_i, f_i = self.create_adaptive_parameter()
        mutant = self.crossover(parents, f_i, cr_i)
        return mutant, cr_i, f_i
        
    def _update(self):
        """Entrena el algoritmo SHADE sobre los datos.

        Args:
            X (np.ndarray): Matriz de datos de entrada.
            y (np.ndarray, optional): Etiquetas reales si están disponibles.
            logger (Any, optional): Objeto para logging del entrenamiento.

        Returns:
            ShadeCC: Instancia entrenada del modelo.
        """
        self._s_cr = np.zeros((0, 0))
        self._sf = np.zeros((0, 0))
        self.fitness_delta = np.zeros((0, 0))

        for current_element in range(self._population_size):
            mutant, cr_i, f_i = self.mutation()
            mutant_fitness = self.fitness(mutant)
            current_fitness = self._population_fitness[current_element]

            if mutant_fitness < current_fitness:
                self._next_population[current_element] = mutant
                self._next_population_fitness[current_element] = mutant_fitness
                self.save_adaptive(current_fitness - mutant_fitness, cr_i, f_i)
            else:
                self._next_population[current_element] = self.population[current_element]
                self._next_population_fitness[current_element] = current_fitness

        self.population = self._next_population
        order = np.argsort(self._next_population_fitness)

        self._population_fitness = self._next_population_fitness[order]
        self.population = self.population[order]

        if len(self._s_cr) > 0 and len(self._sf) > 0:
            self.update_adaptive()

        self._labels = self.get_labels()
        self.get_centroids(self._labels)
