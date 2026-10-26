# enable-shape-type-comments
# Copyright (c) 2005-2024, NumPy Developers. All rights reserved.
# Adapted under LICENSE.numpy; see README.md for source and selected cases.

from typing import Literal, TypeVar

import numpy as np
from numpy._shapes import expand_dims_shape  # noqa: F401

_Scalar = TypeVar("_Scalar", bound=np.generic)

# shapes: *Shape
def expand_dims(
    a: np.ndarray[
        tuple[int, ...],  # shaped: = Shape
        np.dtype[_Scalar],
    ],
    axis: Literal[0],
) -> np.ndarray[
    tuple[int, ...],  # shaped: = expand_dims_shape(Shape, 0)
    np.dtype[_Scalar],
]: ...

# shapes: M, N, R, C
def tile(
    A: np.ndarray[
        tuple[int, int],  # shaped: = [M, N]
        np.dtype[_Scalar],
    ],
    reps: tuple[int, int],  # shaped: = [R, C]
) -> np.ndarray[
    tuple[int, int],  # shaped: = [M * R, N * C]
    np.dtype[_Scalar],
]: ...

# shapes: M, N, P, Q
def kron(
    a: np.ndarray[
        tuple[int, int],  # shaped: = [M, N]
        np.dtype[np.float64],
    ],
    b: np.ndarray[
        tuple[int, int],  # shaped: = [P, Q]
        np.dtype[np.float64],
    ],
) -> np.ndarray[
    tuple[int, int],  # shaped: = [M * P, N * Q]
    np.dtype[np.float64],
]: ...
