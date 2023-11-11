"""
This module contains some methods to calculate the infeasibility of a clustering. The infeasibility is a measure of how
much the clustering violates the constraints. It is one of the most important metrics to evaluate the quality of a
clustering, since it is the only one that takes into account the constraints.
"""

from typing import SupportsIndex, Sequence
import numpy as np


def infeasibility(
    clustering: Sequence[Sequence[SupportsIndex]], constraints: Sequence[Sequence]
) -> float:
    """Infeasibility

    This method calculate the infeasibility of a clustering. The infeasibility is a measure of how much the clustering
    violates the constraints. It is one of the most important metrics to evaluate the quality of a clustering, since it
    is the only one that takes into account the constraints.

    Parameters
    ----------

    clustering: Sequence[Sequence[SupportsIndex]]
        The clustering to evaluate
    constraints: Sequence[Sequence]
        The constraints to evaluate

    Returns
    -------

    float
        The infeasibility of the clustering
    """
    total: int = 0

    for cluster in clustering:
        total += violates_constraints(cluster, constraints)

    return total


def violates_constraints(
    cluster: Sequence[SupportsIndex], constraints: Sequence[Sequence]
) -> int:
    """violates_constraints

    This method calculated the number of constraints violated by a cluster.

    Parameters
    ----------
    cluster: Sequence[SupportsIndex]
        The cluster to evaluate
    constraints: Sequence[Sequence]
        The constraints to evaluate

    Returns
    -------
    int
        The number of constraints violated by the cluster
    """
    linked_mtx = constraints[cluster][:, cluster]
    non_linked_mtx = np.delete(constraints, cluster, axis=0)[:, cluster]

    wrong_links = 0
    for indx in range(len(linked_mtx)):
        wrong_links += np.sum(linked_mtx[indx, indx:] < 0)

    missing_links = np.sum(non_linked_mtx > 0)

    return wrong_links + missing_links
