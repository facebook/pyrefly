import numpy as np
from numpy.typing import ArrayLike, NDArray

from .meshdata import MeshData

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Author: Nicolas P .Rougier
# Date:   04/03/2014
# -----------------------------------------------------------------------------

def create_cube() -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
def create_plane(
    width: float = 1,
    height: float = 1,
    width_segments: int = 1,
    height_segments: float = 1,
    direction: str = "+z",
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
def create_box(
    width: float = 1,
    height: float = 1,
    depth: float = 1,
    width_segments: int = 1,
    height_segments: float = 1,
    depth_segments: float = 1,
    planes: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
def _latitude(rows, cols, radius, offset): ...
def _ico(radius, subdivisions): ...
def _cube(rows, cols, depth, radius): ...
def create_sphere(
    rows: int = 10,
    cols: int = 10,
    depth: int = 10,
    radius: float = 1.0,
    offset: bool = True,
    subdivisions: int = 3,
    method: str = "latitude",
) -> MeshData: ...
def create_cylinder(
    rows: int,
    cols: int,
    radius: tuple[float, ...] = ...,
    length: float = 1.0,
    offset: bool = False,
) -> MeshData: ...
def create_cone(cols: int, radius: float = 1.0, length: float = 1.0) -> MeshData: ...
def create_arrow(
    rows: int,
    cols: int,
    radius: float = 0.1,
    length: float = 1.0,
    cone_radius: float | None = None,
    cone_length: float | None = None,
) -> MeshData: ...
def create_grid_mesh(xs: NDArray, ys: NDArray, zs: NDArray) -> tuple[NDArray, NDArray]: ...
