import os
from os import path as op

import numpy as np
from numpy.typing import ArrayLike

from .stl import load_stl
from .wavefront import WavefrontReader, WavefrontWriter

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

def read_mesh(
    fname: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]: ...
def write_mesh(
    fname: str,
    vertices: ArrayLike,
    faces: ArrayLike | None,
    normals: ArrayLike,
    texcoords: ArrayLike | None,
    name: str = "",
    format: str | None = None,
    overwrite: bool = False,
    reshape_faces: bool = True,
): ...
