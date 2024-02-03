"""Contraint's Matrix

This files contains an implementation of a Contraints Matrix. This is one of the most, if not the most, basic and
frequently used data structures to store the restrictions' information. It is a symmetric matrix (TODO: Add link)
, with as many elements as instances in the dataset, filled in the following way:

$$
    TODO: Add formula for the CM
$$

Note
------

In our library the relationship between a instance and itself is represented as an invalid value that will trigger an
error.
"""
from typing import SupportsIndex, Sequence
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from .._typing import SupportBuffer, ShapeLike, OrderKACF


class ConstraintMatrix:
    """ConstraintMatrix

    This class represent the relationship of a Constraint with the Dataset. Constraints are added knowledge that we
    print in our model. So it specify if two instances can be must-be link (ML) or Can-not link (CL). A relationship
    between pair of instance is represented in each element of the matrix, where values can be set in the range of
    $[-1, 1]$. Negative number indicate a CL relationship and positive ML.

    This class inherit from (np.ndarray)[https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html] since at
    all respect it still being a normal matrix, with the cavenants that the values need to be in the correct range,
    any value out of that will be ... (TODO: TBC).

    Notes
    -------
    This class reimplement the setitem in order to make sure two things:
        - The matrix is symmetric so at any moment $c_ij = c_ji$
        - The values are all in the range of [-1, 1]
    """

    def __init__(self, matrix: Sequence) -> None:
        self.__matrix = np.asarray(matrix)
        self.__ml = {}
        self.__cl = {}

        self.__propagate_constraints()

    def __setitem__(self, index, value):
        if not isinstance(index, SupportsIndex):
            return super().__setitem__(index, value)
        elif not isinstance(index, Sequence):
            return super().__setitem__(index, value)
        elif len(index) != 2:
            return super().__setitem__(index, value)
        else:
            (i, j) = index
            value = int(value)

        # TODO: Should normalize the value to be set following some function
        #       Could be a sigmoid, or another function for some use cases
        #       so guess it should be a parameter of the class
        if value > 1:
            value = 1
        elif value < -1:
            value = -1

        if i != j:
            self.__matrix[i, j] = value
            self.__matrix[j, i] = value

    def __getitem__(self, index):
        self.__matrix[index]

    def __len__(self):
        return self.__matrix.shape[0]

    @property
    def shape(self) -> ShapeLike:
        return self.__matrix.shape

    def __propagate_constraints(self):
        """This method propagate the constraints to create a complete matrix. Constraints are transitive, so if A
        is ML with B, and B is ML with C, then A is ML with C. This method will complete the matrix and raise an
        exception if a contradiction is found.

        Raises
        ------
        ValueError
            If a contradiction is found in the matrix.
        """
        shape = self.__matrix.shape
        must_be_link = {}
        cannot_be_link = {}
        visited = np.zeros(shape[0], dtype=bool)

        def dfs(ml, cl, visited, i, result):
            if visited[i]:
                return
            visited[i] = True

            result[i] = ml[i].copy()

            for j in ml[i]:
                if j in cl[i]:
                    raise ValueError("Contradiction found")

                for k in ml[j]:
                    result[i].add(k)
                    dfs(ml, cl, visited, k, result)

        for i in range(shape[0]):
            if not i in must_be_link:
                must_be_link[i] = set()

            if not i in cannot_be_link:
                cannot_be_link[i] = set()

            for j in range(shape[0]):
                if i == j:
                    continue

                if self.__matrix[i, j] == 0:
                    continue

                if self.__matrix[i, j] > 0:
                    must_be_link[i].add(j)
                    if not j in must_be_link:
                        must_be_link[j] = set()
                    must_be_link[j].add(i)
                    continue

                if self.__matrix[i, j] < 0:
                    cannot_be_link[i].add(j)
                    if not j in cannot_be_link:
                        cannot_be_link[j] = set()
                    cannot_be_link[j].add(i)
                    continue

        # We can't change the dict while iterating over it, so we need to save the result in a new dict
        result = {}

        for i in range(shape[0]):
            dfs(must_be_link, cannot_be_link, visited, i, result)

        must_be_link = result

        for i in range(shape[0]):
            for j in range(shape[0]):
                if i == j:
                    continue

                if j in cannot_be_link[i]:
                    cannot_be_link[i].difference(must_be_link[j])
                    continue

        for i in range(shape[0]):
            for j in range(shape[0]):
                if i == j:
                    continue

                if j in must_be_link[i]:
                    self.__matrix[i, j] = 1
                    self.__matrix[j, i] = 1
                    continue

                if j in cannot_be_link[i]:
                    self.__matrix[i, j] = -1
                    self.__matrix[j, i] = -1
                    continue

                self.__matrix[i, j] = 0
                self.__matrix[j, i] = 0

        self.__ml = must_be_link
        self.__cl = cannot_be_link

    def get_cl_constraints(self, i: int) -> set:
        """This method returns the set of constraints that are cannot-be-link (CL)

        Parameters
        ----------
        i : int
            The index of the instance to get the constraints

        Returns
        -------
        set
            The set of constraints that are cannot-be-link (CL)
        """
        print(f"CL: {self.__cl[i]}")
        return self.__cl[i]

    def get_ml_constraints(self, i: int) -> set:
        """This method returns the set of constraints that are must-be-link (ML)

        Parameters
        ----------
        i : int
            The index of the instance to get the constraints

        Returns
        -------
        set
            The set of constraints that are must-be-link (ML)
        """

        return self.__ml[i]
