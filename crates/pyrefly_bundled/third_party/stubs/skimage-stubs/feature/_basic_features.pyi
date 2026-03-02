from numpy.typing import NDArray

from .._shared import utils

def _texture_filter(gaussian_filtered): ...
def _singlescale_basic_features_singlechannel(img, sigma, intensity=True, edges=True, texture=True): ...
def _mutiscale_basic_features_singlechannel(
    img,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
    num_sigma=None,
    num_workers=None,
): ...
@utils.deprecate_multichannel_kwarg(multichannel_position=1)
def multiscale_basic_features(
    image: NDArray,
    multichannel: bool = False,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    sigma_min: float = 0.5,
    sigma_max: float = 16,
    num_sigma: int | None = None,
    num_workers: int | None = None,
    *,
    channel_axis: int | None = None,
): ...
