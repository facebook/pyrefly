from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage as ndi

from .._shared.utils import _fix_ndimage_mode, _validate_interpolation_order

def profile_line(
    image,
    src,
    dst,
    linewidth: int = 1,
    order=None,
    mode: Literal["constant", "nearest", "reflect", "mirror", "wrap"] = "reflect",
    cval: float = 0.0,
    *,
    reduce_func: Callable = ...,
) -> ArrayLike: ...
def _line_profile_coordinates(src, dst, linewidth=1): ...
