# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_diagonal_shapes() -> None:
    matrix = torch.zeros((4, 5))
    assert_shape(torch.diagonal(matrix).shape, (4,))
    assert_shape(torch.diagonal(matrix, offset=2).shape, (3,))
    assert_shape(matrix.diagonal(offset=-2).shape, (2,))
    assert_shape(torch.diagonal(matrix, offset=6).shape, (0,))

    tensor = torch.zeros((2, 3, 4))
    assert_shape(torch.diagonal(tensor).shape, (4, 2))
    assert_shape(torch.diagonal(tensor, dim1=0, dim2=2).shape, (3, 2))
    assert_shape(tensor.diagonal(offset=1, dim1=1, dim2=2).shape, (2, 3))
    assert_shape(tensor.diagonal(dim1=-2, dim2=-1).shape, (2, 3))


def test_diagonal_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(torch.diagonal(matrix).shape, (2,))

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: diagonal requires at least 2-D input
        torch.diagonal(torch.ones(3))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: diagonal dimensions must be different
        torch.diagonal(matrix, dim1=0, dim2=0)

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: diagonal dim2 out of range
        matrix.diagonal(dim1=0, dim2=2)


def test_diag_embed_shapes() -> None:
    vector = torch.ones(3)
    assert_shape(torch.diag_embed(vector).shape, (3, 3))
    assert_shape(vector.diag_embed().shape, (3, 3))

    matrix = torch.ones((2, 3))
    assert_shape(torch.diag_embed(matrix).shape, (2, 3, 3))
    assert_shape(torch.diag_embed(matrix, offset=2).shape, (2, 5, 5))
    assert_shape(matrix.diag_embed(offset=-2).shape, (2, 5, 5))
    assert_shape(torch.diag_embed(matrix, dim1=0, dim2=1).shape, (3, 3, 2))
    assert_shape(torch.diag_embed(matrix, dim1=2, dim2=0).shape, (3, 2, 3))
    assert_shape(matrix.diag_embed(dim1=-1, dim2=-3).shape, (3, 2, 3))


def test_diag_embed_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(torch.diag_embed(matrix).shape, (2, 3, 3))

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: diag_embed input must have at least one dimension
        torch.diag_embed(torch.ones(()))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: diag_embed dimensions must be different
        torch.diag_embed(matrix, dim1=1, dim2=-2)

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: diag_embed dimension out of range
        torch.diag_embed(matrix, dim1=-4)


if TYPE_CHECKING:

    def check_diagonal[N: IntVar](matrix: Tensor[[N, N]]) -> None:
        assert_type(torch.diagonal(matrix), Tensor[[N]])
        assert_type(matrix.diagonal(), Tensor[[N]])

    def check_diag_embed[B: IntVar, N: IntVar, Offset: IntVar](
        x: Tensor[[B, N]], offset: Int[Offset], dim1: int, dim2: int
    ) -> None:
        assert_type(torch.diag_embed(x), Tensor[[B, N, N]])
        assert_type(torch.diag_embed(x, offset=-2), Tensor[[B, N + 2, N + 2]])
        assert_type(torch.diag_embed(x, offset=offset), Tensor[IntTuple])
        assert_type(torch.diag_embed(x, dim1=dim1, dim2=dim2), Tensor[IntTuple])
