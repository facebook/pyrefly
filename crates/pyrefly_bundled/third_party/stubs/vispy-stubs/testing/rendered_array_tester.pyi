# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

def compare_render(
    orig_data: ArrayLike,
    rendered_data: ArrayLike,
    previous_render: ArrayLike | None = None,
    atol: float | None = 1.0,
): ...
def max_for_dtype(input_dtype: DTypeLike): ...
def make_rgba(data_in: ArrayLike) -> ArrayLike: ...
