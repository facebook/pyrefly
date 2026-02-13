from .._shared import utils

def _hog_normalize_block(block, method, eps=1e-5): ...
def _hog_channel_gradient(channel): ...
@utils.channel_as_last_axis(multichannel_output=False)
@utils.deprecate_multichannel_kwarg(multichannel_position=8)
def hog(
    image,
    orientations: int = 9,
    pixels_per_cell=...,
    cells_per_block=...,
    block_norm="L2-Hys",
    visualize: bool = False,
    transform_sqrt: bool = False,
    feature_vector: bool = True,
    multichannel: bool | None = None,
    *,
    channel_axis: int | None = None,
): ...
