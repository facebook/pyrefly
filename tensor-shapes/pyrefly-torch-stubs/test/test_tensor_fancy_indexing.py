# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the gradual fallback from multi-axis tensor fancy indexing.

Pyrefly does not yet model the broadcast shape of multiple tensor indices.
"""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def test_basic_tensor_index[B: IntVar](z: Tensor[[B, 4, 4]], idx: Tensor[[6]]) -> None:
    """Two tensor indices produce a gradual result shape."""
    result = z[:, idx, idx]
    assert_type(result, Tensor)


def test_slice_and_tensor_index[B: IntVar](
    z: Tensor[[B, 4, 4]], idx: Tensor[[6]]
) -> None:
    """Mixing a slice and tensor index produces a gradual result shape."""
    result = z[:, idx, :]
    assert_type(result, Tensor)


def test_concrete_tensor_index() -> None:
    """Concrete dimensions with tensor indices."""
    z: Tensor[[8, 4, 4]] = torch.randn(8, 4, 4)
    li: Tensor[[6]] = torch.tensor([0, 0, 0, 1, 1, 2])
    lj: Tensor[[6]] = torch.tensor([1, 2, 3, 2, 3, 3])
    result = z[:, li, lj]
    assert_type(result, Tensor)


def test_symbolic_tensor_index[B: IntVar, N: IntVar](
    z: Tensor[[B, 10, 10]], idx: Tensor[[N]]
) -> None:
    """Symbolic index shape."""
    result = z[:, idx, idx]
    assert_type(result, Tensor)
