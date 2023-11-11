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


class ConstraintMatrix(np.ndarray):
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

    def __setitem__(self, index, value):
        (i, j) = index

        # TODO: Should normalize the value to be set following some function
        #       Could be a sigmoid, or another function for some use cases
        #       so guess it should be a parameter of the class
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0

        if i != j:
            super().__setitem__((i, j), value)
            return super().__setitem__((j, i), value)

        return super().__setitem__((i, j), value)

    def __new__(
        cls: type[ArrayLike],
        shape: ShapeLike,
        dtype: DTypeLike = float,
        buffer: None | SupportBuffer = None,
        offset: SupportsIndex = 0,
        strides: None | ShapeLike = None,
        order: OrderKACF = None,
    ) -> ArrayLike:
        if not isinstance(shape, Sequence) and isinstance(shape, SupportsIndex):
            # FIXME: Unsure if this is the best way to check that a shape is a single index
            shape = (shape, shape)

        if len(shape) == 1:
            shape = (shape[0], shape[0])
        if len(shape) != 2:
            raise ValueError("shape should be a only index, or two equals dimensions")
        if shape[0] != shape[1]:
            raise ValueError(
                "shape should be symetrical, so both dimensions need to be the same"
            )

        return super().__new__(
            cls,
            shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )
