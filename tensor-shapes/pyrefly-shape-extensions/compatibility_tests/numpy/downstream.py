# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A consumer of `shaped_library` that never imports `shape_extensions`.

A library's users have not opted into shape types, and their code must keep
checking cleanly. Every checker runs this file. The types each checker infers
differ, so they are pinned separately: `downstream_external.py` for the checkers
that ignore the shape strings, and `downstream_pyrefly.py` for Pyrefly.
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
    Stage,
    transpose,
    untyped_dtype,
)


def annotated_parameters_accept_ordinary_arrays(a: np.ndarray) -> None:
    """A consumer passes plain arrays to `Shaped`-annotated parameters."""

    transpose(a)
    matmul(a, a)
    pad_one(a)
    batched_transpose(a)
    add(a, a)
    untyped_dtype(a)


def annotated_results_support_ordinary_operations(a: np.ndarray) -> None:
    """A `Shaped` return value is an ordinary array to its consumer."""

    result = transpose(a)
    result.reshape(-1)
    result.sum()


def annotated_classes_behave_ordinarily(weight: np.ndarray, x: np.ndarray) -> None:
    """A class declared with `shape_vars` is an ordinary class downstream."""

    encoder = Encoder(weight)
    encoder.weight.sum()
    encoder.encode(x).sum()
    output_size(encoder) + 1


def generic_classes_take_only_their_own_arguments(stage: Stage[str]) -> None:
    """A consumer's `Stage[str]` leaves out the dimension `Stage` declares."""

    assert_type(stage.label, str)
    stage.weight.sum()
