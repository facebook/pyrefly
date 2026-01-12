from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from ..color import Color, get_colormap
from ..color.colormap import CubeHelixColormap
from ..geometry import MeshData
from ..gloo import VertexBuffer
from ..util.event import Event
from .shaders import Function, FunctionChain
from .visual import Visual

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

_VERTEX_SHADER: str = ...

_FRAGMENT_SHADER: str = ...

class MeshVisual(Visual):
    _shaders: dict = ...

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        vertex_colors: ArrayLike | None = None,
        face_colors: ArrayLike | None = None,
        color: Color = ...,
        vertex_values: ArrayLike | None = None,
        meshdata: None | MeshData = None,
        shading: str | None = None,
        mode: str = "triangles",
        **kwargs,
    ): ...
    @property
    def shading(self): ...
    @shading.setter
    def shading(self, shading): ...
    def set_data(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        vertex_colors: ArrayLike | None = None,
        face_colors: ArrayLike | None = None,
        color: Color | None = None,
        vertex_values: ArrayLike | None = None,
        meshdata: None | MeshData = None,
    ): ...
    @property
    def clim(self): ...
    @clim.setter
    def clim(self, clim): ...
    @property
    def _clim_values(self): ...
    @property
    def cmap(self): ...
    @cmap.setter
    def cmap(self, cmap): ...
    @property
    def mode(self): ...
    @mode.setter
    def mode(self, m): ...
    @property
    def mesh_data(self): ...
    @property
    def color(self): ...
    @color.setter
    def color(self, c): ...
    def mesh_data_changed(self): ...
    def _build_color_transform(self, colors): ...
    @staticmethod
    @lru_cache(maxsize=2)
    def _ensure_vec4_func(dims): ...
    def _update_data(self): ...
    def _prepare_draw(self, view): ...
    @staticmethod
    def _prepare_transforms(view): ...
    def _compute_bounds(self, axis, view): ...
