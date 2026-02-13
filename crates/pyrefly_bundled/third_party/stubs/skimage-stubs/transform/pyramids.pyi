import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._shared import utils
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ..transform import resize

def _smooth(image, sigma, mode, cval, channel_axis): ...
def _check_factor(factor): ...
@utils.deprecate_multichannel_kwarg(multichannel_position=6)
def pyramid_reduce(
    image: NDArray,
    downscale: float = 2,
    sigma: float | None = None,
    order: int = 1,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0,
    multichannel: bool = False,
    preserve_range: bool = False,
    *,
    channel_axis: int | None = None,
) -> ArrayLike: ...
@utils.deprecate_multichannel_kwarg(multichannel_position=6)
def pyramid_expand(
    image: NDArray,
    upscale: float = 2,
    sigma: float | None = None,
    order: int = 1,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0,
    multichannel: bool = False,
    preserve_range: bool = False,
    *,
    channel_axis: int | None = None,
) -> ArrayLike: ...
@utils.deprecate_multichannel_kwarg(multichannel_position=7)
def pyramid_gaussian(
    image: NDArray,
    max_layer: int = ...,
    downscale: float = 2,
    sigma: float | None = None,
    order: int = 1,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0,
    multichannel: bool = False,
    preserve_range: bool = False,
    *,
    channel_axis: int | None = None,
): ...
@utils.deprecate_multichannel_kwarg(multichannel_position=7)
def pyramid_laplacian(
    image: NDArray,
    max_layer: int = ...,
    downscale: float = 2,
    sigma: float | None = None,
    order: int = 1,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0,
    multichannel: bool = False,
    preserve_range: bool = False,
    *,
    channel_axis: int | None = None,
): ...
