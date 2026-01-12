import functools
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from .._shared.utils import deprecate_kwarg
from .misc import default_footprint

__all__ = ["erosion", "dilation", "opening", "closing", "white_tophat", "black_tophat"]

def _shift_footprint(footprint, shift_x, shift_y): ...
def _invert_footprint(footprint): ...
def pad_for_eccentric_footprints(func: Callable) -> Callable: ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def erosion(
    image: NDArray,
    footprint: NDArray | None = None,
    out=None,
    shift_x: bool = False,
    shift_y: bool = False,
): ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def dilation(
    image: NDArray,
    footprint: NDArray | None = None,
    out: NDArray | None = None,
    shift_x: bool = False,
    shift_y: bool = False,
): ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
@default_footprint
@pad_for_eccentric_footprints
def opening(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
@default_footprint
@pad_for_eccentric_footprints
def closing(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def white_tophat(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def black_tophat(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
