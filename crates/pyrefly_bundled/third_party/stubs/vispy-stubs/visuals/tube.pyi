import collections

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from ..color import ColorArray
from ..util.svg.color import Color
from ..util.transforms import rotate
from .mesh import MeshVisual

class TubeVisual(MeshVisual):
    def __init__(
        self,
        points: NDArray,
        radius: NDArray | float = 1.0,
        closed: bool = False,
        color: ColorArray | Color | str = "purple",
        tube_points: int = 8,
        shading: str | None = "smooth",
        vertex_colors: NDArray | None = None,
        face_colors: NDArray | None = None,
        mode: str = "triangles",
    ): ...

def _frenet_frames(points, closed): ...
