from numpy.typing import ArrayLike

# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
from ... import glsl
from ...util.svg.transform import Transform
from .raw_point_collection import RawPointCollection

class AggPointCollection(RawPointCollection):
    def __init__(
        self,
        user_dtype: ArrayLike | None = None,
        transform: Transform | None = None,
        vertex: str | None = None,
        fragment: str | None = None,
        **kwargs,
    ): ...
