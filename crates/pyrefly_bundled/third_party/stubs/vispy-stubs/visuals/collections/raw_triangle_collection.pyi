# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import numpy as np
from numpy.typing import NDArray

from ... import glsl
from ..transforms import NullTransform
from .collection import Collection

class RawTriangleCollection(Collection):
    def __init__(self, user_dtype=None, transform=None, vertex=None, fragment=None, **kwargs): ...
    def append(self, points: NDArray, indices: NDArray, **kwargs): ...
