"""Model base
This module contains a BaseEstimator which provides the base's class for the rest of the estimator

Note
----
There is no intention to use this class directly, but to be inherited by other classes. Implementation is based on
scikit-learn's BaseEstimator in order to facilitate the integration with the library.

"""

from abc import ABC
from sklearn.base import BaseEstimator as SklearnBaseEstimator


class BaseEstimator(ABC, SklearnBaseEstimator):
    """
    Base class for estimators in the clustlib package.

    Notes
    -----
    All estimators should specify all the parameters that can be set at the class level in their ``__init__`` as
    explicit keyword arguments (no ``*args`` or ``**kwargs``).

    """
