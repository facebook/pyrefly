# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

import numpy as np

from .normals import normals

def surface(
    func,
    umin=0,
    umax=...,
    ucount=64,
    urepeat=1.0,
    vmin=0,
    vmax=...,
    vcount=64,
    vrepeat=1.0,
): ...
