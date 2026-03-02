import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .._shared.utils import deprecate_kwarg, warn
from . import _util, grayreconstruct

def _add_constant_clip(image, const_value): ...
def _subtract_constant_clip(image, const_value): ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def h_maxima(image: NDArray, h, footprint: NDArray | None = None) -> NDArray: ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def h_minima(image: NDArray, h, footprint: NDArray | None = None) -> NDArray: ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def local_maxima(
    image: NDArray,
    footprint: NDArray | None = None,
    connectivity: int | None = None,
    indices: bool = False,
    allow_borders: bool = True,
) -> np.ndarray | tuple[np.ndarray]: ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def local_minima(
    image: NDArray,
    footprint: NDArray | None = None,
    connectivity: int | None = None,
    indices: bool = False,
    allow_borders: bool = True,
) -> np.ndarray | tuple[np.ndarray]: ...
