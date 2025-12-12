from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require

def _ellipse_in_shape(shape, center, radii, rotation=0.0): ...
def ellipse(r, c, r_radius, c_radius, shape: tuple | None = None, rotation=0.0) -> NDArray: ...
def disk(center: tuple, radius, *, shape: tuple | None = None) -> NDArray: ...
@require("matplotlib", ">=3.0.3")
def polygon_perimeter(r, c, shape: tuple | None = None, clip: bool = False) -> NDArray: ...
def set_color(image, coords, color, alpha=1): ...
def line(r0: int, c0: int, r1: int, c1: int): ...
def line_aa(r0: int, c0: int, r1: int, c1: int): ...
def polygon(r, c, shape: tuple | None = None) -> NDArray: ...
def circle_perimeter(
    r: int,
    c: int,
    radius: int,
    method: Literal["bresenham", "andres"] = "bresenham",
    shape: tuple | None = None,
): ...
def circle_perimeter_aa(r: int, c: int, radius: int, shape: tuple | None = None): ...
def ellipse_perimeter(
    r: int,
    c: int,
    r_radius: int,
    c_radius: int,
    orientation=0,
    shape: tuple | None = None,
): ...
def bezier_curve(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    weight,
    shape: tuple | None = None,
): ...
def rectangle(
    start: tuple,
    end: tuple | None = None,
    extent: tuple | None = None,
    shape: tuple | None = None,
): ...
@require("matplotlib", ">=3.0.3")
def rectangle_perimeter(
    start: tuple,
    end: tuple | None = None,
    extent: tuple | None = None,
    shape: tuple | None = None,
    clip: bool = False,
): ...
def _rectangle_slice(start, end=None, extent=None): ...
