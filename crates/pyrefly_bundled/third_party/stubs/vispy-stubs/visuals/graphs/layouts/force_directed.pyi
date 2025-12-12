import numpy as np
from numpy.typing import ArrayLike

from ..util import _rescale_layout, _straight_line_vertices

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class fruchterman_reingold:
    def __init__(
        self,
        optimal: float | None = None,
        iterations: int = 50,
        pos: ArrayLike | None = None,
    ): ...
    def __call__(self, adjacency_mat: ArrayLike, directed: bool = False): ...
    def _fruchterman_reingold(self, adjacency_mat, directed=False): ...
    def _sparse_fruchterman_reingold(self, adjacency_mat, directed=False): ...

def _calculate_delta_pos(adjacency_arr, pos, t, optimal): ...
