# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""What Pyrefly concludes from the shape strings in `shaped_library`.

Only Pyrefly runs this file. `downstream.py` passes gradual arrays, so it would
pass just as well if Pyrefly ignored the strings; this file would not.
"""

from typing import assert_type

import numpy as np
from shape_extensions import Int
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

F64 = np.dtype[np.float64]


def shapes_flow_through_calls(
    matrix: np.ndarray[[3, 4], F64],
    vector: np.ndarray[[4], F64],
    stack: np.ndarray[[2, 7, 3, 4], F64],
) -> None:
    assert_type(transpose(matrix), np.ndarray[[4, 3], F64])
    assert_type(matmul(matrix, transpose(matrix)), np.ndarray[[3, 3], F64])
    assert_type(pad_one(vector), np.ndarray[[5], F64])
    assert_type(batched_transpose(stack), np.ndarray[[2, 7, 4, 3], F64])
    assert_type(add(matrix, vector), np.ndarray[[3, 4], F64])
    assert_type(untyped_dtype(matrix), np.ndarray[[4, 3]])


def class_declarations_make_the_class_generic(
    weight: np.ndarray[[8, 4], F64], x: np.ndarray[[4], F64]
) -> None:
    encoder = Encoder(weight)
    assert_type(encoder, Encoder[4, 8])
    assert_type(encoder.encode(x), np.ndarray[[8], F64])
    assert_type(output_size(encoder), Int[8])


def mismatched_shapes_are_reported(matrix: np.ndarray[[3, 4], F64]) -> None:
    # `unused-ignore` is an error in this suite, so this fails if the mismatch
    # is no longer reported.
    matmul(matrix, matrix)  # pyrefly: ignore[bad-argument-type]
