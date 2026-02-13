# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ... import gloo, glsl
from ..transforms import NullTransform
from .collection import Collection

class AggFastPathCollection(Collection):
    def __init__(
        self,
        user_dtype: ArrayLike | None = None,
        transform: str | None = None,
        vertex: str | None = None,
        fragment: str | None = None,
        **kwargs,
    ): ...
    def append(self, P: NDArray, closed: bool = False, itemsize: int | None = None, **kwargs): ...
    def bake(self, P, key="curr", closed=False, itemsize=None): ...
    def draw(self, mode="triangle_strip"): ...
