from numpy.typing import NDArray

from ..geometry import create_plane
from ..util.svg.color import Color
from .mesh import MeshVisual
from .visual import CompoundVisual

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

class PlaneVisual(CompoundVisual):
    def __init__(
        self,
        width: float = 1,
        height: float = 1,
        width_segments: int = 1,
        height_segments: float = 1,
        direction: str = "+z",
        vertex_colors: NDArray | None = None,
        face_colors: NDArray | None = None,
        color: Color = ...,
        edge_color: tuple | Color | None = None,
    ): ...
