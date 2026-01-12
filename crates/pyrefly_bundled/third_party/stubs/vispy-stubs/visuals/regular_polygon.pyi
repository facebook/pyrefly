from collections.abc import Sequence

from numpy.typing import ArrayLike

from .ellipse import EllipseVisual

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class RegularPolygonVisual(EllipseVisual):
    def __init__(
        self,
        center: ArrayLike | None = None,
        color: str | tuple | Sequence = "black",
        border_color: str | tuple | Sequence | None = None,
        border_width: float = 1,
        radius: float = 0.1,
        sides: int = 4,
        **kwargs,
    ): ...
    @property
    def sides(self): ...
    @sides.setter
    def sides(self, sides): ...
