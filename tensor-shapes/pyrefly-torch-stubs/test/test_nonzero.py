# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_nonzero_shapes_are_data_dependent() -> None:
    tensor = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    assert_shape(tensor.nonzero().shape, IntTuple, runtime=(2, 2))

    rows, columns = tensor.nonzero(as_tuple=True)
    assert_shape(rows.shape, IntTuple, runtime=(2,))
    assert_shape(columns.shape, IntTuple, runtime=(2,))


if TYPE_CHECKING:

    def check_nonzero_members[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(tensor.nonzero(), Tensor)
        assert_type(tensor.nonzero(as_tuple=True), tuple[Tensor, ...])
