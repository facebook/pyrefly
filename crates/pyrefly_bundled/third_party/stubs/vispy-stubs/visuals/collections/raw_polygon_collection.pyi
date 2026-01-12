# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import numpy as np
from numpy.typing import NDArray

from ... import glsl
from ...geometry import triangulate
from ..transforms import NullTransform
from .collection import Collection

class RawPolygonCollection(Collection):
    def __init__(
        self,
        user_dtype=None,
        transform=None,
        vertex: str | tuple[str, ...] | None = None,
        fragment: str | tuple[str, ...] | None = None,
        **kwargs: str,
    ): ...
    def append(self, points: NDArray, **kwargs): ...
