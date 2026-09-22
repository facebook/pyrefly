# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_einsum_products_and_reordering() -> None:
    left = torch.randn((2, 3))
    right = torch.randn((3, 5))
    tail = torch.randn((5, 7))

    assert_shape(torch.einsum("ij,jk->ik", left, right).shape, (2, 5))
    assert_shape(torch.einsum("ij,jk->ki", left, right).shape, (5, 2))
    assert_shape(torch.einsum("ij,jk,kl->il", left, right, tail).shape, (2, 7))
    assert_shape(torch.einsum("ij->ji", left).shape, (3, 2))


def test_einsum_repeated_labels_and_scalar_output() -> None:
    square = torch.randn((4, 4))
    vector = torch.randn((4,))

    assert_shape(torch.einsum("ii->i", square).shape, (4,))
    assert_shape(torch.einsum("i,i->", vector, vector).shape, ())


def test_einsum_rejects_malformed_equations() -> None:
    left = torch.randn((2, 3))
    right = torch.randn((3, 5))

    with assert_raises(RuntimeError):
        torch.einsum("ij->jk->ik", left, right)  # E: exactly one '->'
    with assert_raises(RuntimeError):
        torch.einsum("ij,!jk->ik", left, right)  # E: unsupported character '!'
    with assert_raises(RuntimeError):
        torch.einsum("ij,jk->ix", left, right)  # E: output index 'x' not found
    with assert_raises(RuntimeError):
        torch.einsum("ij->ii", left)  # E: output index 'i' appears more than once


def test_einsum_rejects_shape_mismatches() -> None:
    left = torch.randn((2, 3))
    wrong = torch.randn((4, 5))

    with assert_raises(RuntimeError):
        torch.einsum("ii->i", left)  # E: conflicting dimensions 2 and 3
    with assert_raises(RuntimeError):
        torch.einsum("ij,jk->ik", left, wrong)  # E: conflicting dimensions 3 and 4
    with assert_raises(RuntimeError):
        torch.einsum("ij,jk->ik", left)  # E: expected 2 operands, got 1
    with assert_raises(RuntimeError):
        torch.einsum("ij,jk->ik", torch.randn((2,)), wrong)  # E: expected rank 2


if TYPE_CHECKING:

    def check_symbolic_einsum[N: IntVar](
        left: Tensor[[N, 2, 3]], right: Tensor[[N, 3, 5]]
    ) -> None:
        assert_type(torch.einsum("bij,bjk->bik", left, right), Tensor[[N, 2, 5]])

    def check_starred_einsum(
        fixed: tuple[Tensor[[2, 3]], Tensor[[3, 5]]],
        homogeneous: list[Tensor[[3, 3]]],
    ) -> None:
        assert_type(torch.einsum("ij,jk->ik", *fixed), Tensor[[2, 5]])
        assert_type(torch.einsum("ij,jk->ik", *homogeneous), Tensor)

    def check_unsupported_einsum(left: Tensor[[2, 3]], right: Tensor[[3, 5]]) -> None:
        assert_type(torch.einsum("ij,jk", left, right), Tensor)
        assert_type(torch.einsum("...ij,...jk->...ik", left, right), Tensor)
