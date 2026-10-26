# enable-shape-type-comments
# Copyright (c) 2024, Joren Hammudoglu (jorenham). All rights reserved.
# Adapted under the BSD license in LICENSE.scipy-stubs.
# NumPy-only excerpt of scipy.fft._helper; see README.md for provenance and scope.

from typing import Literal, TypeVar

import numpy as np

_Scalar = TypeVar("_Scalar", bound=np.generic)

# shapes: N
def fftfreq(
    n: int,  # shaped: N
    d: float = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:  # shaped: .0 = [N]
    ...

# shapes: N
def rfftfreq(
    n: int,  # shaped: N
    d: float = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:  # shaped: .0 = [N // 2 + 1]
    ...

# shapes: *Shape
def fftshift(
    x: np.ndarray[tuple[int, ...], np.dtype[_Scalar]],  # shaped: .0 = Shape
    axes: int | tuple[int, ...] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[_Scalar]]:  # shaped: .0 = Shape
    ...

# shapes: *Shape
def ifftshift(
    x: np.ndarray[tuple[int, ...], np.dtype[_Scalar]],  # shaped: .0 = Shape
    axes: int | tuple[int, ...] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[_Scalar]]:  # shaped: .0 = Shape
    ...
