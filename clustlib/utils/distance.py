from numpy import linalg as la
from numpy import ndarray


def match_distance(name: str):
    """Get the distance function by name.

    Args:
        name (str): The name of the distance function to get.

    Returns:
        DistanceFunction: The distance function.

    """
    if name == "euclidean":
        return euclidean_distance
    else:
        raise ValueError(f"Unknown distance function: {name}")


def euclidean_distance(a: ndarray, **kwargs) -> float | ndarray:
    """Calculate the Euclidean distance between two points.

    Returns:
        float: The Euclidean distance between the two points.

    """
    return la.norm(a, **kwargs)
