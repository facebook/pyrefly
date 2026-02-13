from typing import Literal

from numpy.typing import ArrayLike, NDArray

from .._typing import Scalar
from . import utils

@utils.deprecate_multichannel_kwarg(multichannel_position=5)
def gaussian(
    image: ArrayLike,
    sigma=1,
    output: ArrayLike | None = None,
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "nearest",
    cval: Scalar = 0,
    multichannel=None,
    preserve_range: bool = False,
    truncate: float = 4.0,
    *,
    channel_axis: int | None = None,
) -> NDArray: ...
