# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


def make_pair[D: IntVar](dimension: Int[D]) -> tuple[Tensor[[D]], Tensor[[D, D]]]:
    return torch.randn((dimension,)), torch.randn((dimension, dimension))


vector, matrix = make_pair(4)
assert_type(vector, Tensor[[4]])
assert_type(matrix, Tensor[[4, 4]])

if TYPE_CHECKING:
    assert_type(vector, Tensor[[5]])  # E: assert_type
