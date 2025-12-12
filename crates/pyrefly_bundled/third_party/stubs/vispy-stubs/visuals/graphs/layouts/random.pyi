import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

from ..util import _straight_line_vertices

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

def random(
    adjacency_mat: ArrayLike | NDArray,
    directed: bool = False,
    random_state: None | int | RandomState = None,
): ...
