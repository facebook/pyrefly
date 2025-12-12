from collections.abc import Iterable, Mapping
from typing import Literal

from numpy.typing import NDArray

from .._shared.utils import (
    channel_as_last_axis,
    deprecate_multichannel_kwarg,
)

HOMOGRAPHY_TRANSFORMS = ...

def _preprocess_resize_output_shape(image, output_shape): ...
def resize(
    image: NDArray,
    output_shape: Iterable,
    order=None,
    mode="reflect",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
) -> NDArray: ...
@channel_as_last_axis()
@deprecate_multichannel_kwarg(multichannel_position=7)
def rescale(
    image: NDArray,
    scale: tuple[float, ...] | float,
    order=None,
    mode="reflect",
    cval=0,
    clip=True,
    preserve_range=False,
    multichannel=False,
    anti_aliasing=None,
    anti_aliasing_sigma=None,
    *,
    channel_axis=None,
) -> NDArray: ...
def rotate(
    image: NDArray,
    angle: float,
    resize: bool = False,
    center=None,
    order=None,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
) -> NDArray: ...
def downscale_local_mean(image: NDArray, factors, cval: float = 0, clip: bool = True) -> NDArray: ...
def _swirl_mapping(xy, center, rotation, strength, radius): ...
def swirl(
    image: NDArray,
    center=None,
    strength: float = 1,
    radius: float = 100,
    rotation: float = 0,
    output_shape=None,
    order=None,
    mode="reflect",
    cval=0,
    clip=True,
    preserve_range=False,
) -> NDArray: ...
def _stackcopy(a, b): ...
def warp_coords(coord_map, shape: tuple, dtype=...): ...
def _clip_warp_output(input_image, output_image, mode, cval, clip): ...
def warp(
    image: NDArray,
    inverse_map,
    map_args: Mapping = {},
    output_shape=None,
    order: int | None = None,
    mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
    cval: float = 0.0,
    clip: bool = True,
    preserve_range: bool = False,
): ...
def _linear_polar_mapping(output_coords, k_angle, k_radius, center): ...
def _log_polar_mapping(output_coords, k_angle, k_radius, center): ...
@channel_as_last_axis()
@deprecate_multichannel_kwarg()
def warp_polar(
    image: NDArray,
    center=None,
    *,
    radius: float | None = None,
    output_shape=None,
    scaling: Literal["linear", "log"] = "linear",
    multichannel: bool = False,
    channel_axis: int | None = None,
    **kwargs,
) -> NDArray: ...
def _local_mean_weights(old_size, new_size, grid_mode, dtype): ...
def resize_local_mean(
    image: NDArray, output_shape: Iterable, grid_mode: bool = True, preserve_range: bool = False, *, channel_axis=None
) -> NDArray: ...
