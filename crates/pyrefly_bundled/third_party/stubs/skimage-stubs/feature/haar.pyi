from itertools import chain
from operator import add

import numpy as np
from numpy.typing import ArrayLike

from ..draw import rectangle

FEATURE_TYPE = ...

def _validate_feature_type(feature_type): ...
def haar_like_feature_coord(width: int, height: int, feature_type: str | ArrayLike | None = None): ...
def haar_like_feature(
    int_image,
    r: int,
    c: int,
    width: int,
    height: int,
    feature_type: str | ArrayLike | None = None,
    feature_coord=None,
): ...
def draw_haar_like_feature(
    image,
    r: int,
    c: int,
    width: int,
    height: int,
    feature_coord,
    color_positive_block: tuple[float, float, float] = ...,
    color_negative_block: tuple[float, float, float] = ...,
    alpha: float = 0.5,
    max_n_features: int | None = None,
    random_state=None,
): ...
