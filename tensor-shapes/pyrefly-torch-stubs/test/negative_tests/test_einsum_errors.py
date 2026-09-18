# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Equations `torch.einsum` must reject."""

import torch
from torch import Tensor


# Each annotation quotes the distinctive part of the diagnostic; the shared `einsum` prefix is
# dropped so every call stays on the one line its error is reported against.
def test_malformed_grammar(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    torch.einsum("ij->jk->ik", a, b)  # E: must contain exactly one '->', got 2
    torch.einsum("ij,!jk->ik", a, b)  # E: unsupported character '!' in equation
    torch.einsum("i1,1k->ik", a, b)  # E: unsupported character '1' in equation
    torch.einsum("i..j,jk->ik", a, b)  # E: incomplete ellipsis in equation


def test_missing_output_label(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    torch.einsum("ij,jk->ix", a, b)  # E: output index 'x' not found in inputs


def test_repeated_output_label(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    """A label denotes one extent, so an output term cannot name it twice."""
    torch.einsum("ij->ii", a)  # E: output index 'i' appears more than once
    torch.einsum("ij,jk->ikk", a, b)  # E: output index 'k' appears more than once


def test_ellipsis_does_not_mask_a_malformed_equation(
    a: Tensor[[2, 3]], b: Tensor[[3, 5]]
) -> None:
    """An ellipsis alone is unsupported and silent, but never hides a typo elsewhere."""
    torch.einsum("...ij->...ii", a)  # E: output index 'i' appears more than once
    torch.einsum("...ij,jk->...ix", a, b)  # E: output index 'x' not found in inputs
    torch.einsum("...ij,jk->...i,k", a, b)  # E: unsupported character ',' in equation
    torch.einsum("...ij->...i->j", a)  # E: must contain exactly one '->', got 2


def test_repeated_label_mismatch(a: Tensor[[2, 3]], b: Tensor[[4, 5]]) -> None:
    torch.einsum("ii->i", a)  # E: index 'i' has conflicting dimensions 2 and 3
    torch.einsum("ij,jk->ik", a, b)  # E: index 'j' has conflicting dimensions 3 and 4


def test_operand_count_mismatch(a: Tensor[[2, 3]], b: Tensor[[3, 5]]) -> None:
    torch.einsum("ij,jk->ik", a)  # E: einsum: expected 2 operands, got 1
    torch.einsum("ij->ij", a, b)  # E: einsum: expected 1 operands, got 2


def test_operand_rank_mismatch(
    a: Tensor[[2]], b: Tensor[[3, 5]], c: Tensor[[2, 3, 4]]
) -> None:
    torch.einsum("ij,jk->ik", a, b)  # E: einsum: operand 0 expected rank 2, got 1
    torch.einsum("ij,jk->ik", c, b)  # E: einsum: operand 0 expected rank 2, got 3


def test_no_operands() -> None:
    """An equation always names at least one input, so a bare call cannot satisfy it."""
    torch.einsum("ij->ji")  # E: einsum: expected 1 operands, got 0


def test_non_tensor_operand(a: Tensor[[2, 3]]) -> None:
    torch.einsum("ij,jk->ik", a, 5)  # E: is not assignable to parameter `*operands`


def test_operands_passed_by_keyword(a: Tensor[[2, 3]]) -> None:
    """Operands are variadic, so naming one leaves the equation with nothing to read."""
    # E: Unexpected keyword argument `operands`
    torch.einsum("ij->ji", operands=a)  # E: einsum: expected 1 operands, got 0
