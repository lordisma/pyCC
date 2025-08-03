from typing import Sequence
import numpy as np
from clustlib.gac.base import GeneticClustering
from ..utils.matrix import ConstraintMatrix
import scipy


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
        self.constraints = ConstraintMatrix(constraints)
        self.population_size = population_size
        self.solution_archive = None

    def select_parents(self, fitness, num_parents, ga_instance):
        """Selecciona padres desde la población y el archivo externo.

        Args:
            fitness (np.ndarray): Fitness de cada individuo en la población.
            num_parents (int): Número de padres requeridos.
            ga_instance (GeneticClustering): Instancia de la clase genética.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tupla con los padres seleccionados y sus respectivos fitness.
        """
        fitness_sorted = np.argsort(fitness)
        idx_element = np.random.randint(0, self.population_size)
        x_i = fitness_sorted[idx_element]

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
            x_r2 = fitness_sorted[r2_idx]
            return population[[x_best, x_r1, x_r2]].copy(), np.array(
                [fitness[x_best], fitness[x_r1], fitness[x_r2]]
            )
        else:
            r2_idx -= self.population_size
            if r2_idx > self.solution_archive.shape[0] - 1:
                r2_idx = np.random.randint(0, self.solution_archive.shape[0])

            archive_sol = self.solution_archive[r2_idx]
            fitness_archive = self.fitness(ga_instance, archive_sol, r2_idx)
            return np.ndarray(
                [population[x_i], population[x_best], population[x_r1], archive_sol]
            ), np.array([fitness[x_i], fitness[x_best], fitness[x_r1], fitness_archive])

    def crossover(self, parents, f_i, cr_i):
        """Realiza cruce diferencial entre múltiples padres.

        Args:
            parents (np.ndarray): Soluciones padre [elemento, best, r1, r2].
            f_i (float): Factor de escala.
            cr_i (float): Tasa de cruce.

        Returns:
            np.ndarray: Cromosoma mutante generado.
        """
        element, best, r1, r2 = parents
        mutant = element + f_i * (best - element) + f_i * (r1 - r2)
        mutant = np.clip(mutant, 0, 1)

        cross_points_1 = np.random.rand(len(element)) <= cr_i
        cross_points_2 = np.array(range(len(element))) == np.random.randint(
            len(element), size=len(element)
        )
        cross_points = np.logical_or(cross_points_1, cross_points_2)

        return np.where(cross_points, mutant, element)

    def save_adaptive(self, delta_fitness, cr_i, f_i):
        """Guarda los parámetros exitosos para adaptación futura.

        Args:
            delta_fitness (float): Mejora obtenida en fitness.
            cr_i (float): Tasa de cruce usada.
            f_i (float): Factor de escala usado.
        """
        if self._sf is None:
            self._sf = np.empty((0, 0))
        if self._s_cr is None:
            self._s_cr = np.empty((0, 0))
        if self.fitness_delta is None:
            self.fitness_delta = np.empty((0, 0))

        self._sf = np.append(self._sf, f_i)
        self._s_cr = np.append(self._s_cr, cr_i)
        self.fitness_delta = np.append(self.fitness_delta, delta_fitness)

    def update_adaptive(self, index):
        """Actualiza los historiales H_CR y H_F con medias ponderadas."""
        w_k = self.fitness_delta / self.fitness_delta.sum()
        mean_wa = (w_k * self._s_cr).sum()
        self._h_record_CR = np.append(self._h_record_CR, mean_wa)
        mean_wl = (w_k * (self._sf**2)).sum() / (w_k * self._sf).sum()
        self._h_record_F = np.append(self._h_record_CR, mean_wl)

    def create_adaptive_parameter(self):
        """Genera nuevos parámetros CR y F adaptativos.

        Returns:
            Tuple[float, float]: Par de (CR, F) adaptados.
        """
        r_i = np.random.randint(0, self.population_size)
        while (
            f_i := scipy.stats.cauchy.rvs(
                loc=self._h_record_F[r_i], scale=0.1
            )
        ) <= 0 and f_i > 1.0:
            continue
        while (
            cr_i := np.random.normal(self._h_record_CR[r_i], 0.1)
        ) <= 0 and cr_i > 1.0:
            continue
        return cr_i, f_i

    def mutation(self, current_idx):
        """Genera mutación sobre el individuo actual.

        Args:
            current_idx (int): Índice del individuo actual.

        Returns:
            Tuple[np.ndarray, float, float]: Mutante, cr_i y f_i generados.
        """
        parents, fitness_parents = self.select_parents(
            self._population_fitness, 3, self.ga_instance
        )
        cr_i, f_i = self.create_adaptive_parameter()
        mutant = self.crossover(parents, f_i, cr_i)
        return mutant, cr_i, f_i

    def create_population(self):
        """Inicializa la población y los buffers adaptativos."""
        X_size = self.dataset.shape[0]
        self._external_archive = np.zeros((self.population_size, X_size))
        self._population = np.random.rand(self.population_size, X_size)
        self._h_record_CR = np.full(self.population_size, 0.5)
        self._h_record_F = np.full(self.population_size, 0.5)
        self._population_fitness = np.zeros(self.population_size)
        self._next_population = np.zeros(self.population.shape)
        self._next_population_fitness = np.zeros(self.population_size)

    def fit(self, X, y=None, logger=None):
        """Entrena el algoritmo SHADE sobre los datos.

        Args:
            X (np.ndarray): Matriz de datos de entrada.
            y (np.ndarray, optional): Etiquetas reales si están disponibles.
            logger (Any, optional): Objeto para logging del entrenamiento.

        Returns:
            ShadeCC: Instancia entrenada del modelo.
        """
        num_genes = X.shape[0]
        self.dataset = X
        self.solution_archive = np.zeros((0, num_genes))
        self.create_population()
        self.calculate_fitness()

        for iteration in range(self.max_iter):
            self._s_cr = np.zeros((0, 0))
            self._sf = np.zeros((0, 0))
            self.fitness_delta = np.zeros((0, 0))

            for current_element in range(self.population_size):
                mutant, cr_i, f_i = self.mutation(current_element)
                mutant_fitness = self.fitness(mutant)
                current_fitness = self._population_fitness[current_element]

                if mutant_fitness < current_fitness:
                    self._next_population[current_element] = mutant
                    self._next_population_fitness[current_element] = mutant_fitness
                    self.save_adaptive(current_fitness - mutant_fitness, cr_i, f_i)
                else:
                    self._next_population[current_element] = self._population[
                        current_element
                    ]
                    self._next_population_fitness[current_element] = current_fitness

            self._population = self._next_population
            self._population_fitness = self._next_population_fitness
            self._population_fitness_sorted = np.argsort(current_fitness)

            if len(self._s_cr) > 0 and len(self._sf) > 0:
                self.update_adaptive(iteration % self.population_size)

        labels = self.get_labels()
        self.get_centroids(labels)
        return self
