import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from .._shared.utils import deprecate_kwarg
from .misc import default_footprint

# The default_footprint decorator provides a diamond footprint as
# default with the same dimension as the input image and size 3 along each
# axis.
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def binary_erosion(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def binary_dilation(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None): ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def binary_opening(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None) -> NDArray: ...
@default_footprint
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def binary_closing(image: NDArray, footprint: NDArray | None = None, out: NDArray | None = None) -> NDArray: ...
