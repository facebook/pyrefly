# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""What a checker that ignores the shape strings concludes.

Pyrefly skips this file, because it reads the strings and infers shaped types.

Because the carrier form wraps numpy's own shape argument, these checkers still
see numpy's rank and dtype, such as `ndarray[tuple[int, int], dtype[float64]]`.
`untyped_dtype` uses the shorthand instead, and degrades to a bare array.

`assert_type` fails both if the shape metadata leaks into a type and if a type
collapses to `Any`.
"""

from typing import assert_type

import numpy as np
from shaped_library import (
    add,
    batched_transpose,
    Encoder,
    matmul,
    output_size,
    pad_one,
    transpose,
    untyped_dtype,
)

Rank1 = np.ndarray[tuple[int], np.dtype[np.float64]]
Rank2 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
AnyRank = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def the_carrier_survives_as_numpys_own_shape_type(a: np.ndarray, b: np.ndarray) -> None:
    assert_type(transpose(a), Rank2)
    assert_type(matmul(a, b), Rank2)
    assert_type(pad_one(a), Rank1)
    assert_type(batched_transpose(a), AnyRank)
    assert_type(add(a, b), AnyRank)


def the_shorthand_degrades_to_a_bare_array(a: np.ndarray) -> None:
    assert_type(untyped_dtype(a), np.ndarray)


def annotated_attributes_and_methods_keep_their_rank(
    weight: np.ndarray, x: np.ndarray
) -> None:
    encoder = Encoder(weight)
    assert_type(encoder.weight, Rank2)
    assert_type(encoder.encode(x), Rank1)
    assert_type(output_size(encoder), int)
