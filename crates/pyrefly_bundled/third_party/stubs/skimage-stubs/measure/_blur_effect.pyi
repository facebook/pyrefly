from typing import Callable

from numpy.typing import NDArray

__all__ = ["blur_effect"]

def blur_effect(
    image: NDArray,
    h_size: int = 11,
    channel_axis: int | None = None,
    reduce_func: Callable = ...,
): ...
