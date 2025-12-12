import numpy as np
from numpy.typing import ArrayLike

from ..color import ColorArray
from ..util.svg.color import Color
from .visual import Visual

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

_VERTEX_SHADER: str = ...

_FRAGMENT_SHADER: str = ...

class WindbarbVisual(Visual):
    _shaders: dict = ...

    def __init__(self, **kwargs): ...
    def set_data(
        self,
        pos: ArrayLike | None = None,
        wind: ArrayLike | None = None,
        trig: bool = True,
        size: float | ArrayLike = 50.0,
        antialias: float = 1.0,
        edge_width: None | float = 1.0,
        edge_color: ColorArray | Color | str = "black",
        face_color: ColorArray | Color | str = "white",
    ): ...
    def _prepare_transforms(self, view): ...
    def _prepare_draw(self, view): ...
    def _compute_bounds(self, axis, view): ...
