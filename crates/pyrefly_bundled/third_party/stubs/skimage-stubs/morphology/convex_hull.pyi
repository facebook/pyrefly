from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._shared.utils import warn
from ..measure._label import label
from ..measure.pnpoly import grid_points_in_poly

__all__ = ["convex_hull_image", "convex_hull_object"]

def _offsets_diamond(ndim): ...
def _check_coords_in_hull(gridcoords, hull_equations, tolerance): ...
def convex_hull_image(image: ArrayLike, offset_coordinates: bool = True, tolerance: float = 1e-10): ...
def convex_hull_object(image, *, connectivity=2) -> NDArray: ...
