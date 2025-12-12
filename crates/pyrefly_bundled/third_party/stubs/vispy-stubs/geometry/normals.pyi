import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

def compact(vertices, indices, tolerance=1e-3): ...
def normals(vertices: NDArray, indices: NDArray): ...
