import re

import numpy as np
from numpy.typing import ArrayLike

from ..util import logger
from .buffer import DataBuffer, IndexBuffer, VertexBuffer
from .context import get_current_canvas
from .globject import GLObject
from .preprocessor import preprocess
from .texture import BaseTexture, Texture1D, Texture2D, Texture3D, TextureCube
from .util import check_enum

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

# ------------------------------------------------------------ Constants ---
REGEX_VAR: dict = ...

# ------------------------------------------------------------ Shader class ---
class Shader(GLObject):
    def __init__(self, code=None): ...
    @property
    def code(self): ...
    @code.setter
    def code(self, code): ...

class VertexShader(Shader):
    _GLIR_TYPE: str = ...

class FragmentShader(Shader):
    _GLIR_TYPE: str = ...

class GeometryShader(Shader):
    _GLIR_TYPE: str = ...

# ----------------------------------------------------------- Program class ---
class Program(GLObject):
    _GLIR_TYPE: str = ...

    _gtypes: dict = ...

    # ---------------------------------
    def __init__(self, vert: str | None = None, frag: str | None = None, count: int | None = 0): ...
    def set_shaders(
        self,
        vert: str,
        frag: str,
        geom: str | None = None,
        update_variables: bool = True,
    ): ...
    @property
    def shaders(self): ...
    @property
    def variables(self) -> ArrayLike: ...
    def _parse_variables_from_code(self, update_variables=True): ...
    def bind(self, data: VertexBuffer): ...
    def _process_pending_variables(self): ...
    def __setitem__(self, name, data): ...
    def __contains__(self, key): ...
    def __getitem__(self, name): ...
    def draw(
        self,
        mode: str = "triangles",
        indices: ArrayLike | None = None,
        check_error=True,
    ): ...
