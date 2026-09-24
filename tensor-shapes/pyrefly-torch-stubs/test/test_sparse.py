# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_sparse_shapes() -> None:
    tensor = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    sparse = tensor.to_sparse()
    assert_shape(sparse.shape, (2, 3))
    assert_shape(sparse.indices().shape, IntTuple, runtime=(2, 2))


if TYPE_CHECKING:

    def check_sparse_members[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(tensor.to_sparse(), Tensor[Shape])
        assert_type(tensor.to_sparse().indices(), Tensor)
