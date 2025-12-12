import numpy as np
from numpy.typing import ArrayLike

from ..util.svg.color import Color
from .polygon import PolygonVisual

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class EllipseVisual(PolygonVisual):
    def __init__(
        self,
        center: ArrayLike | None = None,
        color: Color | str = "black",
        border_color: Color | None = None,
        border_width: float = 1,
        radius: float | tuple = ...,
        start_angle: float = 0.0,
        span_angle: float = 360.0,
        num_segments: int = 100,
        **kwargs,
    ): ...
    @staticmethod
    def _generate_vertices(center, radius, start_angle, span_angle, num_segments): ...
    @property
    def center(self): ...
    @center.setter
    def center(self, center): ...
    @property
    def radius(self): ...
    @radius.setter
    def radius(self, radius): ...
    @property
    def start_angle(self): ...
    @start_angle.setter
    def start_angle(self, start_angle): ...
    @property
    def span_angle(self): ...
    @span_angle.setter
    def span_angle(self, span_angle): ...
    @property
    def num_segments(self): ...
    @num_segments.setter
    def num_segments(self, num_segments): ...
    def _regen_pos(self): ...
