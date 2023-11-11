"""typing

This module define some utilities types used on algorithms.

"""

from typing import SupportsIndex, Sequence, Type, Any, Union, Literal
from numpy import array, memmap, ndarray, dtype, generic

ShapeLike = Union[Type[SupportsIndex], Type[Sequence[SupportsIndex]]]

SupportBuffer = Union[
    Type[bytes],
    Type[bytearray],
    Type[memoryview],
    Type[array],
    Type[memmap],
    Type[ndarray[Any, dtype[Any]]],
    Type[generic],
]

"""OrderKACF

Numpy orders supported, this are straight literals to indicated the accepted values
"""
OrderKACF = Literal[None, "K", "A", "C", "F"]
