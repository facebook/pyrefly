import numpy as np
from numpy.typing import NDArray

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

###############################################################################
# These fast normal calculation routines are adapted from mne-python

def _fast_cross_3d(x, y): ...
def _calculate_normals(rr, tris): ...
def resize(image: NDArray, shape: tuple, kind: str = "linear") -> NDArray: ...
