# enable-shape-type-comments
# Copyright (c) 2005-2024, NumPy Developers. All rights reserved.
# Adapted under LICENSE.numpy; see README.md for source and selected cases.

from typing import Literal, overload, TypeVar

import numpy as np

_Scalar = TypeVar("_Scalar", bound=np.generic)

# shapes: N
@overload
def diag(
    v: np.ndarray[
        tuple[int],  # shaped: type = [N]
        np.dtype[_Scalar],
    ],
    k: Literal[0] = 0,
) -> np.ndarray[
    tuple[int, int],  # shaped: type = [N, N]
    np.dtype[_Scalar],
]: ...

# shapes: N
@overload
def diag(
    v: np.ndarray[
        tuple[int, int],  # shaped: type = [N, N]
        np.dtype[_Scalar],
    ],
    k: Literal[0] = 0,
) -> np.ndarray[
    tuple[int],  # shaped: type = [N]
    np.dtype[_Scalar],
]: ...

# shapes: *Batch, M, N
def tril(
    v: np.ndarray[
        tuple[int, ...],  # shaped: type = [*Batch, M, N]
        np.dtype[_Scalar],
    ],
    k: int = 0,
) -> np.ndarray[
    tuple[int, ...],  # shaped: type = [*Batch, M, N]
    np.dtype[_Scalar],
]: ...

# shapes: *Batch, M, N
def triu(
    v: np.ndarray[
        tuple[int, ...],  # shaped: type = [*Batch, M, N]
        np.dtype[_Scalar],
    ],
    k: int = 0,
) -> np.ndarray[
    tuple[int, ...],  # shaped: type = [*Batch, M, N]
    np.dtype[_Scalar],
]: ...
