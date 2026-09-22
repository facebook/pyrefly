# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import (
    assert_raises,
    assert_shape,
    Elements,
    Int,
    IntTuple,
    IntVar,
)
from torch import Tensor


def test_integer_and_slice_indexing() -> None:
    tensor = torch.randn((10, 20, 30))
    assert_shape(tensor[:5].shape, (5, 20, 30))
    assert_shape(tensor[0:3].shape, (3, 20, 30))
    assert_shape(tensor[0].shape, (20, 30))
    assert_shape(tensor[:, -1, :].shape, (10, 30))
    assert_shape(tensor[:, :, :].shape, (10, 20, 30))
    assert_shape(tensor[:-1].shape, (9, 20, 30))
    assert_shape(tensor[-2:].shape, (2, 20, 30))
    assert_shape(tensor[::2].shape, (5, 20, 30))
    assert_shape(tensor[1:9:3].shape, (3, 20, 30))


def test_new_axis_indexing() -> None:
    tensor = torch.randn((5, 10))
    assert_shape(tensor[None].shape, (1, 5, 10))
    assert_shape(tensor[None, 0].shape, (1, 10))
    assert_shape(tensor[:, None, :].shape, (5, 1, 10))
    assert_shape(tensor[..., None].shape, (5, 10, 1))


def test_ellipsis_indexing() -> None:
    tensor = torch.randn((2, 3, 4, 5, 6))
    assert_shape(tensor[..., 0].shape, (2, 3, 4, 5))
    assert_shape(tensor[..., 0, 0].shape, (2, 3, 4))
    assert_shape(tensor[0, ..., 0].shape, (3, 4, 5))


def test_index_out_of_bounds() -> None:
    tensor = torch.randn((2, 3))
    assert_shape(tensor.shape, (2, 3))
    with assert_raises(IndexError):
        # TODO: BUG: Reject out-of-bounds literal indices statically.
        tensor[2]


def test_parameter_slice() -> None:
    parameter = torch.nn.Parameter(torch.randn((10, 20)))
    assert_shape(parameter[:5].shape, (5, 20))
    assert_shape(torch.nn.Parameter(parameter[:5]).shape, (5, 20))


if TYPE_CHECKING:

    def check_symbolic_indexing[B: IntVar, T: IntVar, V: IntVar](
        tensor: Tensor[[B, T, V]],
    ) -> None:
        assert_type(tensor[:, 0, :], Tensor[[B, V]])
        assert_type(tensor[:, -1, :], Tensor[[B, V]])
        assert_type(tensor[:, :, :], Tensor[[B, T, V]])
        assert_type(tensor[3:], Tensor[[B - 3, T, V]])
        assert_type(tensor[::2], Tensor[[(B + 1) // 2, T, V]])

    def check_bare_tensor(tensor: Tensor) -> None:
        assert_type(tensor[0], Tensor)
        assert_type(tensor[:], Tensor)
        assert_type(tensor[:, -1, :], Tensor)

    def check_ellipsis[B: IntVar, T: IntVar, N: IntVar, D: IntVar](
        tensor: Tensor[[B, T, N, D, 2]],
    ) -> None:
        assert_type(tensor[..., 0], Tensor[[B, T, N, D]])
        assert_type(tensor[..., 0, 0], Tensor[[B, T, N]])

    def check_variadic_indexing[B: IntVar, D: IntVar, Shape: IntTuple, C: IntVar](
        tensor: Tensor[[B, D, *Elements[Shape], C]],
    ) -> None:
        assert_type(tensor[0, :], Tensor[[D, *Elements[Shape], C]])
        assert_type(tensor[:, 0], Tensor[[B, *Elements[Shape], C]])
        assert_type(tensor[..., 0], Tensor[[B, D, *Elements[Shape]]])
        assert_type(tensor[0, ...], Tensor[[D, *Elements[Shape], C]])

    def check_invalid_index(tensor: Tensor[[10, 20]]) -> None:
        tensor["bad"]  # E: Cannot index into

    def check_shape_tuple_slicing[B: IntVar, T: IntVar, N: IntVar, D: IntVar](
        tensor: Tensor[[B, T, N, D]],
    ) -> None:
        assert_type(tensor.size()[:-1], tuple[Int[B], Int[T], Int[N]])
        assert_type(
            tensor.float().reshape(*tensor.size()[:-1], -1, 2),
            Tensor[[B, T, N, D // 2, 2]],
        )
