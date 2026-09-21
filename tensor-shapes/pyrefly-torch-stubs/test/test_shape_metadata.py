# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, cast, Literal, TYPE_CHECKING

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


def test_shape_and_size() -> None:
    scalar = torch.randn(())
    assert_shape(scalar.shape, ())
    assert_type(scalar.shape, tuple[()])
    assert_type(scalar.size(), tuple[()])

    tensor = torch.randn((2, 3, 4, 5, 6))
    assert_shape(tensor.shape, (2, 3, 4, 5, 6))
    assert_type(
        tensor.shape,
        tuple[Literal[2], Literal[3], Literal[4], Literal[5], Literal[6]],
    )
    assert_type(
        tensor.size(),
        tuple[Literal[2], Literal[3], Literal[4], Literal[5], Literal[6]],
    )


def test_size_dimension() -> None:
    tensor = torch.randn((2, 7, 4))
    assert_shape(tensor.shape, (2, 7, 4))
    assert_type(tensor.size(0), Literal[2])
    assert_type(tensor.size(1), Literal[7])
    assert_type(tensor.size(-1), Literal[4])


def test_size_rejects_invalid_dimensions() -> None:
    tensor = torch.randn((2, 3))
    with assert_raises(IndexError):
        tensor.size(2)  # E: size dimension out of range
    with assert_raises(IndexError):
        tensor.size(-3)  # E: size dimension out of range

    scalar = torch.randn(())
    with assert_raises(IndexError):
        scalar.size(0)  # E: size dimension out of range
    with assert_raises(IndexError):
        scalar.size(-1)  # E: size dimension out of range


def test_element_count_and_rank() -> None:
    scalar = torch.randn(())
    assert_shape(scalar.shape, ())
    assert_type(scalar.numel(), Literal[1])
    assert_type(scalar.nelement(), Literal[1])
    assert_type(scalar.dim(), Literal[0])

    empty = torch.empty((2, 0, 3))
    assert_shape(empty.shape, (2, 0, 3))
    assert_type(empty.numel(), Literal[0])
    assert_type(torch.numel(empty), Literal[0])
    assert_type(empty.nelement(), Literal[0])
    assert_type(empty.dim(), Literal[3])

    tensor = torch.randn((3, 4, 5))
    assert_shape(tensor.shape, (3, 4, 5))
    assert_type(tensor.numel(), Literal[60])
    assert_type(torch.numel(tensor), Literal[60])
    assert_type(tensor.nelement(), Literal[60])
    assert_type(tensor.dim(), Literal[3])
    assert_type(tensor.dim() + 2, Literal[5])


def test_tolist_is_gradual() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]])
    assert_shape(tensor.shape, (2, 2))
    assert_type(tensor.tolist(), Any)
    assert tensor.tolist() == [[1, 2], [3, 4]]


def test_variadic_shape_from_linear() -> None:
    output = torch.nn.Linear(10, 20)(torch.randn((5, 10)))
    assert_shape(output.shape, (5, 20))
    assert_type(output.size(-1), Int[20])
    assert_type(output.size(0), Int[5])


if TYPE_CHECKING:

    def check_shape_metadata[N: IntVar, M: IntVar, Shape: IntTuple](
        symbolic: Tensor[[N, M]],
        mixed: Tensor[[N, 3, 4]],
        arithmetic: Tensor[[N + 1, N * 2]],
        gradual_element: Tensor[[int, 3]],
        gradual_rank: Tensor[IntTuple],
        unpacked_rank: Tensor[[*Elements[Shape]]],
        bare: Tensor,
        dynamic_dimension: int,
    ) -> None:
        assert_type(symbolic.shape, tuple[Int[N], Int[M]])
        assert_type(mixed.shape, tuple[Int[N], Literal[3], Literal[4]])
        assert_type(arithmetic.shape, tuple[Int[N + 1], Int[N * 2]])
        assert_type(symbolic.size(), tuple[Int[N], Int[M]])
        assert_type(symbolic.size(-1), Int[M])
        assert_type(symbolic.numel(), Int[N * M])

        assert_type(gradual_element.numel(), Int[int])
        assert_type(gradual_rank.numel(), Int[int])
        assert_type(unpacked_rank.numel(), Int[int])
        assert_type(symbolic.dim(), Literal[2])
        assert_type(unpacked_rank.dim(), Int[int])
        assert_type(gradual_rank.dim(), Int[int])

        assert_type(bare.shape, IntTuple)
        assert_type(bare.size(), IntTuple)
        assert_type(bare.size(0), Int[int])
        assert_type(bare.size(dynamic_dimension), Int[int])
        assert_type(bare.numel(), Int[int])

    def check_size_preserves_shape[Shape: IntTuple](
        tensor: Tensor[Shape], scalar: Tensor[[]]
    ) -> None:
        assert_type(tensor.size(), Shape)
        assert_type(scalar.size(), tuple[()])

    def check_variadic_suffix[Batch: IntTuple, Out: IntVar](
        tensor: Tensor[[*Elements[Batch], Out]],
    ) -> None:
        assert_type(tensor.size(-1), Int[Out])
        assert_type(tensor.size(0), Int[int])

    assert_type(cast(Tensor[[2, 3]], ...).numel(), Literal[6])
