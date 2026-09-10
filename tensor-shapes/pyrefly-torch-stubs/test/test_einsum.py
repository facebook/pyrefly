# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shape inference for `torch.einsum` over the explicit equation subset."""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def test_matrix_product(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    assert_type(torch.einsum("ij,jk->ik", a, b), Tensor[[2, 5]])


def test_output_order(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    """Output dimensions follow the order the output term spells, not input order."""
    assert_type(torch.einsum("ij,jk->ki", a, b), Tensor[[5, 2]])


def test_whitespace_is_insignificant(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    assert_type(torch.einsum(" i j , j k -> i k ", a, b), Tensor[[2, 5]])


def test_scalar_output(a: Tensor[[4]], b: Tensor[[4]]) -> None:
    assert_type(torch.einsum("i,i->", a, b), Tensor[[]])


def test_transpose(a: Tensor[[2, 3]]) -> None:
    assert_type(torch.einsum("ij->ji", a), Tensor[[3, 2]])


def test_three_operands(
    a: Tensor[[2, 3]], b: Tensor[[3, 5]], c: Tensor[[5, 7]]
) -> None:
    assert_type(torch.einsum("ij,jk,kl->il", a, b, c), Tensor[[2, 7]])


def test_repeated_label_equal(a: Tensor[[4, 4]]) -> None:
    """Occurrences that agree keep the dimension precise."""
    assert_type(torch.einsum("ii->i", a), Tensor[[4]])


def test_symbolic_labels[N: IntVar](a: Tensor[[N, 2, 3]], b: Tensor[[N, 3, 5]]) -> None:
    assert_type(torch.einsum("bij,bjk->bik", a, b), Tensor[[N, 2, 5]])


def test_repeated_label_symbolic_equal[N: IntVar](a: Tensor[[N, N]]) -> None:
    """Occurrences spelled the same denote one extent, so precision survives."""
    assert_type(torch.einsum("ii->i", a), Tensor[[N]])


def test_repeated_label_literal_constrains_symbolic[N: IntVar](
    a: Tensor[[N, 3]],
) -> None:
    """A consistent literal occurrence determines the repeated label's extent."""
    assert_type(torch.einsum("ii->i", a), Tensor[[3]])


def test_repeated_label_symbolic_unequal[N: IntVar, M: IntVar](
    a: Tensor[[N, M]],
) -> None:
    """Occurrences that cannot be shown equal widen rather than picking the first.

    Only the dimension they reach widens: the output rank is fixed by the equation.
    """
    assert_type(torch.einsum("ii->i", a), Tensor[[int]])


def test_gradual_operand(a: Tensor[[2, 3]], b: Tensor) -> None:
    """A gradual operand widens only the dimensions read from it."""
    assert_type(torch.einsum("ij,jk->ik", a, b), Tensor[[2, int]])


def test_fixed_starred_operands(
    operands: tuple[Tensor[[2, 3]], Tensor[[3, 5]]],
) -> None:
    """Unpacking a fixed-length tuple keeps every operand shape distinct."""
    assert_type(torch.einsum("ij,jk->ik", *operands), Tensor[[2, 5]])


def test_homogeneous_starred_operands(operands: list[Tensor[[3, 3]]]) -> None:
    """An unknown operand count cannot establish that every equation input is present."""
    assert_type(torch.einsum("ij,jk->ik", *operands), Tensor)


def test_homogeneous_starred_gradual_operands(operands: list[Tensor]) -> None:
    assert_type(torch.einsum("ij,jk->ik", *operands), Tensor)


def test_implicit_output_is_unsupported(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    """An implicit output equation is valid but unmodelled, so it falls back silently."""
    assert_type(torch.einsum("ij,jk", a, b), Tensor)


def test_ellipsis_is_unsupported(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    assert_type(torch.einsum("...ij,...jk->...ik", a, b), Tensor)


def test_well_formed_ellipsis_stays_unsupported(a: Tensor[[3, 3]]) -> None:
    """Checking the rest of an ellipsis equation must not turn a valid one into an error."""
    assert_type(torch.einsum("...ii->...i", a), Tensor)
