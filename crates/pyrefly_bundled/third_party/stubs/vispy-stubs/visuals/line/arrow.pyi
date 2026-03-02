import numpy as np
from numpy.typing import ArrayLike

from ... import gloo, glsl
from ...util.svg.color import Color
from ..transforms._util import as_vec4
from ..visual import Visual
from .line import LineVisual

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

ARROW_TYPES = ...

class _ArrowHeadVisual(Visual):
    ARROWHEAD_VERTEX_SHADER = ...
    ARROWHEAD_FRAGMENT_SHADER = ...

    _arrow_vtype = ...

    def __init__(self, parent: ArrowVisual): ...
    def _prepare_transforms(self, view): ...
    def _prepare_draw(self, view=None): ...
    def _prepare_vertex_data(self): ...

class ArrowVisual(LineVisual):
    def __init__(
        self,
        pos: ArrayLike | None = None,
        color: Color | tuple | ArrayLike = ...,
        width=1,
        connect: str | ArrayLike = "strip",
        method: str = "gl",
        antialias: bool = False,
        arrows: ArrayLike | None = None,
        arrow_type: str = "stealth",
        arrow_size: float | None = None,
        arrow_color: Color | tuple | ArrayLike = ...,
    ): ...
    def set_data(
        self,
        pos: ArrayLike | None = None,
        color: Color | tuple | ArrayLike | None = None,
        width=None,
        connect: str | ArrayLike | None = None,
        arrows: ArrayLike | None = None,
    ): ...
    @property
    def arrow_type(self): ...
    @arrow_type.setter
    def arrow_type(self, value): ...
    @property
    def arrow_size(self): ...
    @arrow_size.setter
    def arrow_size(self, value): ...
    @property
    def arrow_color(self): ...
    @arrow_color.setter
    def arrow_color(self, value): ...
    @property
    def arrows(self): ...
