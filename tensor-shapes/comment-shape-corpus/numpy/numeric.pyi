# enable-shape-type-comments
# Copyright (c) 2005-2024, NumPy Developers. All rights reserved.
# Adapted under LICENSE.numpy; see README.md for source and selected cases.

from typing import Literal, TypeVar

import numpy as np

_Scalar = TypeVar("_Scalar", bound=np.generic)

# shapes: M, N
def outer(
    a: np.ndarray[
        tuple[int],  # shaped: type = [M]
        np.dtype[np.float64],
    ],
    b: np.ndarray[
        tuple[int],  # shaped: type = [N]
        np.dtype[np.float64],
    ],
    out: (
        np.ndarray[
            tuple[int, int],  # shaped: type = [M, N]
            np.dtype[np.float64],
        ]
        | None
    ) = None,
) -> np.ndarray[
    tuple[int, int],  # shaped: type = [M, N]
    np.dtype[np.float64],
]: ...

# shapes: M, N, P
def tensordot(
    a: np.ndarray[
        tuple[int, int],  # shaped: type = [M, N]
        np.dtype[np.float64],
    ],
    b: np.ndarray[
        tuple[int, int],  # shaped: type = [N, P]
        np.dtype[np.float64],
    ],
    axes: Literal[1],
) -> np.ndarray[
    tuple[int, int],  # shaped: type = [M, P]
    np.dtype[np.float64],
]: ...

# shapes: *Shape
def roll(
    a: np.ndarray[tuple[int, ...], np.dtype[_Scalar]],  # shaped: .args[0] = Shape
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[_Scalar]]:  # shaped: .args[0] = Shape
    ...
