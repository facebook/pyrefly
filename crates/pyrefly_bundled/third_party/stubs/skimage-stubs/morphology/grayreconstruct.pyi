from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._shared.utils import deprecate_kwarg

@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def reconstruction(
    seed: NDArray,
    mask: NDArray,
    method: Literal["erosion", "dilation"] = "dilation",
    footprint: NDArray | None = None,
    offset: NDArray | None = None,
) -> NDArray: ...
