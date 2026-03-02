from .._shared.utils import deprecate_multichannel_kwarg

def _generate_rectangle_mask(point, image, shape, random): ...
def _generate_circle_mask(point, image, shape, random): ...
def _generate_triangle_mask(point, image, shape, random): ...
def _generate_ellipse_mask(point, image, shape, random): ...

# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = ...
SHAPE_CHOICES = ...

def _generate_random_colors(num_colors, num_channels, intensity_range, random): ...
@deprecate_multichannel_kwarg(multichannel_position=5)
def random_shapes(
    image_shape: tuple,
    max_shapes: int,
    min_shapes: int = 1,
    min_size: int = 2,
    max_size: int | None = None,
    multichannel: bool = True,
    num_channels: int = 3,
    shape=None,
    intensity_range=None,
    allow_overlap: bool = False,
    num_trials: int = 100,
    random_seed=None,
    *,
    channel_axis: int | None = ...,
): ...
