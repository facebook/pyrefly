# enable-shape-type-comments
# Copyright (c) 2024, Joren Hammudoglu (jorenham). All rights reserved.
# Adapted under the BSD license in LICENSE.scipy-stubs.
# NumPy-only excerpt of scipy.linalg._decomp_cholesky; see README.md.

from typing import overload

import numpy as np

type _LowerFlag = bool | np.ndarray[tuple[()], np.dtype[np.bool]]

# shapes: *Batch, N
def cholesky(
    a: np.ndarray[
        tuple[int, ...],  # shaped: = [*Batch, N, N]
        np.dtype[np.float64],
    ],
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> np.ndarray[
    tuple[int, ...],  # shaped: = [*Batch, N, N]
    np.dtype[np.float64],
]: ...

# shapes: *Batch, N
def cho_factor(
    a: np.ndarray[
        tuple[int, ...],  # shaped: = [*Batch, N, N]
        np.dtype[np.float64],
    ],
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[
    np.ndarray[
        tuple[int, ...],  # shaped: = [*Batch, N, N]
        np.dtype[np.float64],
    ],
    np.ndarray[
        tuple[int, ...],  # shaped: = Batch
        np.dtype[np.bool],
    ],
]: ...
@overload
# shapes: N
def cho_solve(
    c_and_lower: tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],  # shaped: [0] = [N, N]
        _LowerFlag,
    ],
    b: np.ndarray[tuple[int], np.dtype[np.float64]],  # shaped: [0] = [N]
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:  # shaped: [0] = [N]
    ...
@overload
# shapes: N, K
def cho_solve(
    c_and_lower: tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],  # shaped: [0] = [N, N]
        _LowerFlag,
    ],
    b: np.ndarray[tuple[int, int], np.dtype[np.float64]],  # shaped: [0] = [N, K]
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:  # shaped: [0] = [N, K]
    ...
