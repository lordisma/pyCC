"""Model base
This module contains a BaseEstimator which provides the base's class for the rest of the estimator

There is no intention to use this class directly, but to be inherited by other classes. Implementation is based on
scikit-learn's BaseEstimator in order to facilitate the integration with the library.
"""

from abc import ABC
from ._typing import InitCentroid
from sklearn.base import ClusterMixin as SklearnBaseEstimator

import numpy as np

from .utils.simpleconstraints import SimpleConstraints
from .utils.initilize import random, kmeans

import logging

logger = logging.getLogger(__name__)

class BaseEstimator(ABC, SklearnBaseEstimator):
    """
    Base class for estimators in the clustlib package.

    Attributes:
        labels_ (numpy.ndarray): Labels of the dataset.

    Notes:
        All estimators should specify all the parameters that can be set at the class level in their `__init__` 
        as explicit keyword arguments (no `*args` or `**kwargs`).
    """
    centroids: np.ndarray = None
    init: InitCentroid
    
    n_clusters: int
    tol: float
    max_iter: int = None
    
    constraints: SimpleConstraints
    X: np.ndarray
    _labels: np.array = None
    
    _delta: np.ndarray = None

    def fit(self, dataset: np.ndarray, labels: np.array = None):
        self.X = np.copy(dataset)

        if self.centroids is None:
            if isinstance(self.init, np.ndarray):
                self.centroids = self.init
            elif self.init == "random":
                self.centroids = random(dataset, self.n_clusters)
            elif self.init == "kmeans":
                self.centroids = kmeans(dataset, self.n_clusters)
            else:
                raise ValueError(f"Unknown initialization method")
            
        if self._labels is None:
            self._labels = np.random.randint(0, self.n_clusters, dataset.shape[0], dtype=int)
        else:
            self._labels = np.copy(labels)
        
        return self._fit()

    def _fit(self):
        """ Fit the model to the data.

        Args:
            dataset (numpy.ndarray): The data to cluster.
            labels (numpy.ndarray, optional): Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns:
            BaseEstimator: The fitted estimator.
        """
        raise NotImplementedError
    
    def predict(self, x: np.array) -> int:
        """
        Predict the cluster index for a given instance.

        Args:
            x (numpy.ndarray): The instance to be predicted.

        Returns:
            int: The index of the cluster to which the instance is assigned.
        """
        return np.argmin(np.linalg.norm(self.centroids - x))
    
    def calculte_delta(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the difference between the new and old centroids.

        This method is used to determine when the algorithm has converged.

        Args:
            x (numpy.ndarray): The old centroids.

        Returns:
            numpy.ndarray: The absolute difference between the new and old centroids.
        """
        return np.abs(self.centroids - x)
    
    def update(self):
        """
        Update the centroids of the clusters.

        This method calls the `_update` method to update the centroids of the clusters. It also updates the `_delta` 
        attribute with the difference between the new and old centroids. The `_delta` attribute is a numpy array 
        with the same shape as the centroids and is used to determine when the algorithm has converged.
        """
        aux = np.copy(self.centroids)
        self._update()
        self._delta = self.calculte_delta(aux)

    def _update(self):
        """Update the centroids of the clusters.

        This method should be implemented by any class that inherits from it.
        """
        raise NotImplementedError

    def _convergence(self):
        """
        Check convergence of the algorithm.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        if self._delta is None:
            logger.debug("Delta is None, convergence cannot be checked.")
            return False
    
    def stop_criteria(self, iteration) -> bool:
        """
        Check if the algorithm has reached the stopping criteria.

        Args:
            iteration (int): The current iteration of the algorithm.

        Returns:
            bool: True if the algorithm has reached the stopping criteria, False otherwise.
        """
        if self._convergence():
            logger.debug("Convergence reached, stopping criteria met.")
            return True

        if self.max_iter is None:
            logger.debug("No maximum iterations set, stopping criteria not met.")
            return False
        
        return iteration > self.max_iter
