# enable-shape-type-comments
# Copyright (c) 2024, Joren Hammudoglu (jorenham). All rights reserved.
# Adapted under the BSD license in LICENSE.scipy-stubs.
# NumPy-only excerpt of scipy.spatial.distance; see README.md.

from typing import Literal

import numpy as np
import numpy.typing as npt

# shapes: M, P, N
def cdist(
    XA: np.ndarray[
        tuple[int, int],  # shaped: = [M, N]
        np.dtype[np.float64],
    ],
    XB: np.ndarray[
        tuple[int, int],  # shaped: = [P, N]
        np.dtype[np.float64],
    ],
    metric: Literal["euclidean"] = "euclidean",
    *,
    out: (
        np.ndarray[
            tuple[int, int],  # shaped: = [M, P]
            np.dtype[np.float64],
        ]
        | None
    ) = None,
) -> np.ndarray[
    tuple[int, int],  # shaped: = [M, P]
    np.dtype[np.float64],
]: ...

# shapes: M, N
def pdist(
    X: npt.NDArray[np.float64],  # shaped: = np.ndarray[[M, N], np.dtype[np.float64]]
    metric: Literal["euclidean"] = "euclidean",
    *,
    out: (
        np.ndarray[
            tuple[int],  # shaped: = [M * (M - 1) // 2]
            np.dtype[np.float64],
        ]
        | None
    ) = None,
) -> np.ndarray[
    tuple[int],  # shaped: = [M * (M - 1) // 2]
    np.dtype[np.float64],
]: ...
