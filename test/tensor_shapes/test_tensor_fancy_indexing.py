# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for multi-axis tensor fancy indexing.

When multiple tensor indices appear in a multi-axis index (e.g. Z[:, li, lj]),
the indexed dims are replaced by the broadcast shape of the tensor indices.
"""

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_basic_tensor_index[B](z: Tensor[B, 4, 4], idx: Tensor[6]) -> None:
    """Two tensor indices replace dims 1 and 2 with the index shape."""
    result = z[:, idx, idx]
    assert_type(result, Tensor[B, 6])


def test_slice_and_tensor_index[B](z: Tensor[B, 4, 4], idx: Tensor[6]) -> None:
    """Slice preserves dim, tensor index replaces."""
    result = z[:, idx, :]
    assert_type(result, Tensor[B, 6, 4])


def test_concrete_tensor_index() -> None:
    """Concrete dimensions with tensor indices."""
    z: Tensor[8, 4, 4] = torch.randn(8, 4, 4)
    li: Tensor[6] = torch.tensor([0, 0, 0, 1, 1, 2])
    lj: Tensor[6] = torch.tensor([1, 2, 3, 2, 3, 3])
    result = z[:, li, lj]
    assert_type(result, Tensor[8, 6])


def test_symbolic_tensor_index[B, N](z: Tensor[B, 10, 10], idx: Tensor[N]) -> None:
    """Symbolic index shape."""
    result = z[:, idx, idx]
    assert_type(result, Tensor[B, N])
