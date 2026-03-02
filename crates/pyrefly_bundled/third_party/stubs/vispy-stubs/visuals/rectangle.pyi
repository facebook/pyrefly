import numpy as np
from numpy.typing import ArrayLike

from ..color import Color
from .polygon import PolygonVisual

# Copradiusight (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class RectangleVisual(PolygonVisual):
    def __init__(
        self,
        center: ArrayLike | None = None,
        color: Color | str = "black",
        border_color: Color | None = None,
        border_width: int = 1,
        height: float = 1.0,
        width: float = 1.0,
        radius: ArrayLike | float = [0.0, 0.0, 0.0, 0.0],
        **kwargs,
    ): ...
    @staticmethod
    def _generate_vertices(center, radius, height, width): ...
    @property
    def center(self): ...
    @center.setter
    def center(self, center): ...
    @property
    def height(self): ...
    @height.setter
    def height(self, height): ...
    @property
    def width(self): ...
    @width.setter
    def width(self, width): ...
    @property
    def radius(self): ...
    @radius.setter
    def radius(self, radius): ...
    def _regen_pos(self): ...
