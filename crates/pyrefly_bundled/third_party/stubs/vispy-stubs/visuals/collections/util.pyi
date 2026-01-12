#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (c) 2013, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------

from functools import reduce
from operator import mul

import numpy as np

def dtype_reduce(dtype, level=0, depth=0): ...
def fetchcode(utype, prefix=""): ...
