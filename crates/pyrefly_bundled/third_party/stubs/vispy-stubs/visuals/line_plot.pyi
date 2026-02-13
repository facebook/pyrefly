import numpy as np
from numpy.typing import ArrayLike

from ..util.svg.color import Color
from .line import LineVisual
from .markers import MarkersVisual
from .visual import CompoundVisual

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class LinePlotVisual(CompoundVisual):
    _line_kwargs = ...
    _marker_kwargs = ...
    _valid_kwargs = ...
    _kw_trans = ...

    def __init__(
        self,
        data: ArrayLike | None = None,
        color: Color | str = "k",
        symbol: str | None = None,
        line_kind: str = "-",
        width: float = 1.0,
        marker_size: float = 10.0,
        edge_color: Color | str = "k",
        face_color: Color | str = "w",
        edge_width: float = 1.0,
        connect: str | ArrayLike = "strip",
    ): ...
    def set_data(self, data: ArrayLike | None = None, **kwargs): ...
