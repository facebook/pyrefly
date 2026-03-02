from numpy.typing import ArrayLike, NDArray

from ..color import Color
from ..geometry.isosurface import isosurface
from .mesh import MeshVisual

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class IsosurfaceVisual(MeshVisual):
    def __init__(
        self,
        data: NDArray | None = None,
        level: None | float = None,
        vertex_colors: NDArray | None = None,
        face_colors: NDArray | None = None,
        color: NDArray | None = ...,
        **kwargs,
    ): ...
    @property
    def level(self): ...
    @level.setter
    def level(self, level): ...
    def set_data(
        self,
        data: NDArray | None = None,
        vertex_colors: ArrayLike | None = None,
        face_colors: ArrayLike | None = None,
        color: Color | None = None,
    ): ...
    def _prepare_draw(self, view): ...
