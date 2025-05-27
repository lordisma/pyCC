"""Model base
This module contains a BaseEstimator which provides the base's class for the rest of the estimator

Note
----
There is no intention to use this class directly, but to be inherited by other classes. Implementation is based on
scikit-learn's BaseEstimator in order to facilitate the integration with the library.

"""

from abc import ABC
from ._typing import InitCentroid
from sklearn.base import ClusterMixin as SklearnBaseEstimator

import numpy as np

from .utils.matrix import ConstraintMatrix
from .utils.initilize import random, kmeans


class BaseEstimator(ABC, SklearnBaseEstimator):
    """
    Base class for estimators in the clustlib package.

    Attributes
    __________

    labels_: numpy.ndarray
        Labels of the dataset

    Notes
    -----
    All estimators should specify all the parameters that can be set at the class level in their ``__init__`` as
    explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """
    centroids: np.ndarray
    init: InitCentroid
    
    n_clusters: int
    tol: float
    max_iter: int
    
    constraints: ConstraintMatrix
    X: np.ndarray
    _labels: np.array
    
    __delta: np.ndarray

    def fit(self, dataset: np.ndarray, constraints: np.ndarray, labels: np.array = None):
        self.constraints = constraints

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
        
        return self.fit(dataset, labels)

    def fit(self, dataset, labels=None):
        """Fit the model to the data.

        Parameters
        __________
        dataset: numpy.ndarray
            The data to cluster.
        labels: numpy.ndarray, default=None
            Ignored. This parameter exists only for compatibility with the sklearn API.

        Returns
        _______
        self
            The fitted estimator.
        """
        raise NotImplementedError
    
    def predict(self, x: np.array) -> int:
        """Check if the labels for the better fit of the instance and predict the
        data based on that.

        Parameters
        ____________
        x: numpy.array
            The instance to be predicted

        Returns
        ________
        cluster_index: int
            The index of the cluster at which the instance was assigned
        """
        return np.argmin(np.linalg.norm(self.centroids - x))
    
    def calculte_delta(self, x: np.array) -> np.ndarray:
        """Calculate the difference between the new and old centroids.

        This method is used to determine when the algorithm has reached an end.

        Parameters
        __________
        x: numpy.array
            The old centroids
        """
        if self.__delta is None:
            return np.zeros(self.centroids.shape)
        
        return self.centroids - x
    
    def update(self):
        """Update the centroids of the clusters

        This method will call the _update method to update the centroids of the clusters.
        It will also update the _delta_centroid attribute with the difference between the new and old centroids.
        The _delta attribute is a numpy array with the same shape as the centroids and used to determine
        when the algorithm has reached an end.
        """
        aux = np.copy(self.centroids)
        self._update()
        self.__delta = self.calculte_delta(aux)
    
    def _update(self):
        """Update the centroids of the clusters.

        This method should be implemented by any class that inherits from it.
        """
        raise NotImplementedError

    def _convergence(self):
        """Check convergence of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has converged, False otherwise.
        """
        if self.__delta is None:
            return False
        
        return np.all(np.linalg.norm(self.__delta, axis=1) <= self.tol)
    
    def stop_criteria(self, iteration):
        """Check if the algorithm has reached the maximum number of iterations.

        Parameters
        __________
        iteration: int
            The current iteration of the algorithm.

        Returns
        _______
        bool
            True if the algorithm has reached the maximum number of iterations, False otherwise.
        """

        if self._convergence():
            return True
        
        if self.max_iter is None or self.max_iter <= 0:
            return False
        
        return iteration >= self.max_iter
