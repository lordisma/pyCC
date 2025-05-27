from numpy import linalg as la
from numpy import ndarray
from typing import Callable, Dict

type DistanceFunction = Callable[[ndarray, ndarray, Dict], float]

def match_distance(name: str) -> DistanceFunction:
    """Get the distance function by name.

    Parameters
    __________
    name: str
        The name of the distance function to get.

    Returns
    _______
    DistanceFunction
        The distance function.
    """
    if name == "euclidean":
        return euclidean_distance
    else:
        raise ValueError(f"Unknown distance function: {name}")
    
def euclidean_distance(a: ndarray, kwargs: Dict) -> float:
    """Calculate the Euclidean distance between two points.

    Parameters
    __________
    a: numpy.ndarray
        The first point.
    b: numpy.ndarray
        The second point.

    Returns
    _______
    float
        The Euclidean distance between the two points.
    """
    return la.norm(a, **kwargs)