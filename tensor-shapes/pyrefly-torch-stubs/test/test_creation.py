# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, Literal, TYPE_CHECKING

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


def test_zeros() -> None:
    assert_shape(torch.zeros((2, 3)).shape, (2, 3))
    with assert_raises(TypeError):
        torch.zeros("invalid")  # E: No matching overload


def test_size_factories() -> None:
    assert_shape(torch.zeros(2, 3).shape, (2, 3))
    assert_shape(torch.ones((2, 3)).shape, (2, 3))
    assert_shape(torch.empty((2, 0, 3)).shape, (2, 0, 3))
    assert_shape(torch.full((2, 3), 1.0).shape, (2, 3))
    assert_shape(torch.rand(2, 3, dtype=None).shape, (2, 3))
    assert_shape(torch.randn(()).shape, ())
    with assert_raises(RuntimeError):
        # TODO: BUG: Reject negative literal dimensions statically.
        torch.zeros((-1, 2))


if TYPE_CHECKING:

    def check_size_factories[N: IntVar](
        n: Int[N],
        plain: int,
        unbounded: tuple[int, ...],
        unpacked: tuple[Literal[1], *tuple[int, ...], Literal[3]],
        dimensions: list[int],
    ) -> None:
        assert_type(torch.randn(n, plain), Tensor[[N, int]])
        assert_type(torch.rand((n, plain)), Tensor[[N, int]])
        assert_type(torch.zeros(*dimensions), Tensor[IntTuple])
        assert_type(torch.ones(unbounded), Tensor[IntTuple])
        assert_type(torch.empty(*unpacked), Tensor[[1, *Elements[IntTuple], 3]])
        assert_type(torch.full(unbounded, 1.0), Tensor[IntTuple])

    def check_like_factories[Shape: IntTuple](x: Tensor[Shape]) -> None:
        assert_type(torch.zeros_like(x), Tensor[Shape])
        assert_type(torch.ones_like(x), Tensor[Shape])
        assert_type(torch.empty_like(x), Tensor[Shape])
        assert_type(torch.full_like(x, 2.5), Tensor[Shape])
        assert_type(torch.rand_like(x), Tensor[Shape])
        assert_type(torch.randn_like(x), Tensor[Shape])


def test_like_factories() -> None:
    x = torch.randn((2, 0, 3))
    for result in (
        torch.zeros_like(x),
        torch.ones_like(x),
        torch.empty_like(x),
        torch.full_like(x, 2.5),
        torch.rand_like(x),
        torch.randn_like(x),
    ):
        assert_shape(result.shape, (2, 0, 3))

    scalar = torch.tensor(1.0)
    for result in (
        torch.zeros_like(scalar),
        torch.ones_like(scalar),
        torch.empty_like(scalar),
        torch.full_like(scalar, 2.5),
        torch.rand_like(scalar),
        torch.randn_like(scalar),
    ):
        assert_shape(result.shape, ())
