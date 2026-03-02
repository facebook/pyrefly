from collections.abc import Iterable, Mapping

import numpy as np

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.#!/usr/bin/env python3
from ..util import _straight_line_vertices

class NetworkxCoordinates:
    def __init__(self, graph=None, layout: str | dict | Iterable[np.float32] | None = None, **kwargs: Mapping): ...
    def __call__(self, adjacency_mat, directed: bool = False): ...
    @property
    def adj(self): ...
