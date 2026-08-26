# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_type, Literal, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_tuple_and_list_indices() -> None:
    tensor = torch.randn((10, 20, 30))
    assert_shape(tensor[:, (0, 2, 4), :].shape, (10, 3, 30))
    assert_shape(tensor[0, (1, 2), :].shape, (2, 30))
    assert_shape(tensor[(0, 1), :10, 5].shape, (2, 10))

    # A list's runtime length is not part of its static type.
    assert_shape(tensor[:, [0, 2], :].shape, (10, int, 30), runtime=(10, 2, 30))


def test_tensor_indices() -> None:
    tensor = torch.randn((8, 4, 4))
    rows = torch.tensor([0, 0, 0, 1, 1, 2])
    columns = torch.tensor([1, 2, 3, 2, 3, 3])

    # TODO: BUG: Preserve the broadcast shape of tensor indices.
    assert_shape(tensor[:, rows, columns].shape, IntTuple, runtime=(8, 6))
    assert_shape(tensor[:, rows, :].shape, IntTuple, runtime=(8, 6, 4))


def test_boolean_index() -> None:
    tensor = torch.arange(12).reshape(3, 4)
    mask = torch.tensor([True, False, True])
    assert_shape(tensor[mask].shape, IntTuple, runtime=(2, 4))


if TYPE_CHECKING:

    def check_typed_indices[N: IntVar](
        tensor: Tensor[[10, 20, 30]],
        bound: Int[N],
        dynamic_bound: int,
        indices: tuple[int, int, int],
        list_indices: list[int],
    ) -> None:
        assert_type(tensor[:bound], Tensor[[N, 20, 30]])
        assert_type(tensor[:dynamic_bound], Tensor[[Any, 20, 30]])
        assert_type(tensor[:, indices, :], Tensor[[10, 3, 30]])
        assert_type(tensor[:, list_indices, :], Tensor[[10, Any, 30]])

    def check_literal_bound(tensor: Tensor[[10, 20, 30]], bound: Literal[5]) -> None:
        assert_type(tensor[:bound], Tensor[[5, 20, 30]])

    def check_gradual_index_domain(
        tensor: Tensor[[8, 4, 4]],
        index: Tensor[[2]],
        sequence: Sequence[int],
        nested: list[list[int]],
    ) -> None:
        assert_type(tensor[True], Tensor)
        assert_type(tensor[:, index, (0, 1)], Tensor)
        assert_type(tensor[sequence], Tensor)
        assert_type(tensor[nested], Tensor)
