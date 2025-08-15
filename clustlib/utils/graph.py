"""Constraints Graph

This module contains the implementation of the constraints graph. This graph is a representation of the constraints
where each node will represent an instance of the dataset and the edges will represent the relationship between two
instances. The edges will be weighted with the value of the constraint, where a positive value will indicate a must-link
relationship and a negative value a cannot-link relationship.
"""

from typing import Sequence, SupportsIndex

import numpy as np

from .matrix import ConstraintMatrix


class ConstraintsGraph:
    """Constraints Graph

    This class represent the graph of constraints. It is a representation of the constraints where each node will
    be an instance of the dataset and the edges will represent the relationship between two instances.

    Methods
    -------
    add_constraint(i, j, value)
        Add a constraint to the graph
    """

    __adj_matrix: ConstraintMatrix

    def __init__(self, number_intances: int, constraints: Sequence[Sequence]):
        """Constructor

        The Constraints Graph is constructed from the size of the dataset and the constraints.
        The size is needed to know the instances that will be represented in the graph and could do assumptions about
        cardinality and density. The constraints are the restrictions that will be represented with the edges of the
        graph.

        Parameters
        ----------

        n: int
            The number of instances in the dataset
        constraints: Sequence[Sequence]
            The constraints that will be represented in the graph
        """

        self.__adj_matrix = np.zeros((number_intances, number_intances))

        for i, j, value in constraints:
            self.add_constraint(i, j, value)

    def add_constraint(self, i: SupportsIndex, j: SupportsIndex, value: float):
        """Add a constraint to the graph

        This method add a constraint to the graph. It will add the constraint to the matrix and will make sure that the
        matrix is symmetric.

        Parameters
        ----------

        i: SupportsIndex
            The index of the first instance in the constraint
        j: SupportsIndex
            The index of the second instance in the constraint
        value: float
            The value of the constraint
        """
        self.__adj_matrix[i, j] = value

    def get_constraint(self, i: SupportsIndex, j: SupportsIndex) -> float:
        """Get the constraint value

        This method return the value of the constraint between the two instances. If the value is not set, it will
        return 0.0

        Parameters
        ----------
        i: SupportsIndex
            The index of the first instance in the constraint
        j: SupportsIndex
            The index of the second instance in the constraint

        Returns
        -------
        float
            The value of the constraint
        """
        return self.__adj_matrix[i, j]
