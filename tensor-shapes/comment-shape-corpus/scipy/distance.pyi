# enable-shape-type-comments
# Copyright (c) 2024, Joren Hammudoglu (jorenham). All rights reserved.
# Adapted under the BSD license in LICENSE.scipy-stubs.
# NumPy-only excerpt of scipy.spatial.distance; see README.md.

from typing import Literal

import numpy as np
import numpy.typing as npt

# shapes: M, P, N
def cdist(
    XA: np.ndarray[tuple[int, int], np.dtype[np.float64]],  # shaped: .args[0] = [M, N]
    XB: np.ndarray[tuple[int, int], np.dtype[np.float64]],  # shaped: .args[0] = [P, N]
    metric: Literal["euclidean"] = "euclidean",
    *,
    out: npt.NDArray[np.float64]
    | None = None,  # shaped: type = np.ndarray[[M, P], np.dtype[np.float64]] | None
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:  # shaped: .args[0] = [M, P]
    ...

# shapes: M, N
def pdist(
    X: npt.NDArray[
        np.float64
    ],  # shaped: type = np.ndarray[[M, N], np.dtype[np.float64]]
    metric: Literal["euclidean"] = "euclidean",
    *,
    out: npt.NDArray[np.float64]
    | None = None,  # shaped: type = np.ndarray[[M * (M - 1) // 2], np.dtype[np.float64]] | None
) -> np.ndarray[
    tuple[int], np.dtype[np.float64]
]:  # shaped: .args[0] = [M * (M - 1) // 2]
    ...
