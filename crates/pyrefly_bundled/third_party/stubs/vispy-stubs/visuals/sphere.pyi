from numpy.typing import NDArray

from ..geometry import create_sphere
from ..util.svg.color import Color
from .mesh import MeshVisual
from .visual import CompoundVisual

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

class SphereVisual(CompoundVisual):
    def __init__(
        self,
        radius: float = 1.0,
        cols: int = 30,
        rows: int = 30,
        depth: int = 30,
        subdivisions: int = 3,
        method: str = "latitude",
        vertex_colors: NDArray | None = None,
        face_colors: NDArray | None = None,
        color: Color = ...,
        edge_color: tuple | Color | None = None,
        shading: str | None = None,
        **kwargs,
    ): ...
    @property
    def mesh(self): ...
    @property
    def border(self): ...
