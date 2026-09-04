# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""What Pyrefly concludes from `shaped_library` against real numpy.

Only the real-numpy Pyrefly check runs this file. Real numpy's shape parameter
is a `tuple`, which an `IntTuple` sits on top of, so both the carrier form and
the shorthand carry a shape.
"""

from typing import assert_type, Literal

import numpy as np
from shape_extensions import IntTuple
from shaped_library import pad_one, transpose, untyped_dtype

F64 = np.dtype[np.float64]


def carriers_carry_shapes(
    matrix: np.ndarray[tuple[Literal[3], Literal[4]], F64],
    vector: np.ndarray[tuple[int], F64],
) -> None:
    assert_type(transpose(matrix), np.ndarray[IntTuple[4, 3], F64])
    assert_type(pad_one(vector), np.ndarray[IntTuple[int], F64])


def the_shorthand_carries_shapes(
    matrix: np.ndarray[tuple[Literal[3], Literal[4]], F64],
) -> None:
    assert_type(untyped_dtype(matrix), np.ndarray[IntTuple[4, 3]])
