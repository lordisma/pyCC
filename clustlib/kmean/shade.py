from typing import Sequence
import numpy as np
from model import BaseEstimator
from ..utils.matrix import ConstraintMatrix

class ShadeCC(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        init="random",
        max_iter=300,
        tol=1e-4,
        custom_initial_centroids=None,
        constraints: Sequence[Sequence] = None,
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

    def fit(self, dataset, labels=None):
        raise NotImplementedError
    
    def fitness(self, dataset, centroids):
        raise NotImplementedError
    
    def local_search(self):
        raise NotImplementedError
    
    def search_for_movement(self):
        raise NotImplementedError

    