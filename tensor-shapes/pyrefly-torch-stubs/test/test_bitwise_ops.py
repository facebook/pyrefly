# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test bitwise operators on boolean and integer tensors"""

from typing import assert_type, TYPE_CHECKING

from shape_extensions import IntVar


if TYPE_CHECKING:
    from torch import Tensor


def test_and_same_shape[N: IntVar, M: IntVar](x: Tensor[[N, M]], y: Tensor[[N, M]]):
    """Combining two masks preserves the shape"""
    assert_type(x & y, Tensor[[N, M]])


def test_xor[N: IntVar, M: IntVar](x: Tensor[[N, M]], y: Tensor[[N, M]]):
    """`^` behaves like `&`"""
    assert_type(x ^ y, Tensor[[N, M]])


def test_or_between_two_tensors_is_read_as_a_union[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], y: Tensor[[N, M]]
):
    """`|` between two tensors resolves to a union type instead of calling `__or__`.

    This is a checker bug rather than a gap in the stub: `x.__or__(y)` and
    `x | True` both give `Tensor[[N, M]]`, and `&` and `^` are unaffected. The
    PEP 604 union reading only wins when both operands are tensors. The
    assertion below records the wrong answer so the test suite stays green and
    the day this is fixed is visible.
    """
    assert_type(x.__or__(y), Tensor[[N, M]])
    assert_type(x | y, type[Tensor[[N, M]]])


def test_invert[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """Negating a mask preserves the shape"""
    assert_type(~x, Tensor[[N, M]])


def test_broadcast[N: IntVar, M: IntVar](x: Tensor[[N, M]], y: Tensor[[1, M]]):
    """Bitwise operators broadcast the way the comparison operators do"""
    assert_type(x & y, Tensor[[N, M]])


def test_scalar_operand[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """A scalar operand on either side preserves the shape"""
    assert_type(x & True, Tensor[[N, M]])
    assert_type(True & x, Tensor[[N, M]])


def test_mask_pipeline[N: IntVar, M: IntVar](x: Tensor[[N, M]], y: Tensor[[N, M]]):
    """The motivating case: masks produced by comparisons, combined with `&`"""
    mask = (x != 0) & (y != 0)
    assert_type(mask, Tensor[[N, M]])
