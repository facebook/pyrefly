# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test newly migrated identity operations using Self return types."""

from typing import assert_type, cast

from torch import Tensor


def test_trig_functions() -> None:
    """Test trigonometric functions preserve shape via Self."""
    x = cast(Tensor[2, 3], ...)

    # Test newly added trig functions
    assert_type(x.asin(), Tensor[2, 3])
    assert_type(x.acos(), Tensor[2, 3])
    assert_type(x.atan(), Tensor[2, 3])
    assert_type(x.sinh(), Tensor[2, 3])
    assert_type(x.cosh(), Tensor[2, 3])


def test_math_functions() -> None:
    """Test mathematical functions preserve shape via Self."""
    x = cast(Tensor[4, 5], ...)

    # Test newly added math functions
    assert_type(x.exp2(), Tensor[4, 5])
    assert_type(x.expm1(), Tensor[4, 5])
    assert_type(x.log2(), Tensor[4, 5])
    assert_type(x.log10(), Tensor[4, 5])
    assert_type(x.log1p(), Tensor[4, 5])
    assert_type(x.rsqrt(), Tensor[4, 5])
    assert_type(x.square(), Tensor[4, 5])
    assert_type(x.reciprocal(), Tensor[4, 5])
    assert_type(x.sign(), Tensor[4, 5])
    assert_type(x.sigmoid(), Tensor[4, 5])
    assert_type(x.trunc(), Tensor[4, 5])
    assert_type(x.frac(), Tensor[4, 5])


def test_special_functions() -> None:
    """Test special functions preserve shape via Self."""
    x = cast(Tensor[3, 3], ...)

    # Test error functions
    assert_type(x.erfinv(), Tensor[3, 3])

    # Test gamma functions
    assert_type(x.lgamma(), Tensor[3, 3])
    assert_type(x.digamma(), Tensor[3, 3])
    assert_type(x.polygamma(2), Tensor[3, 3])

    # Test inverse hyperbolic
    assert_type(x.asinh(), Tensor[3, 3])
    assert_type(x.acosh(), Tensor[3, 3])
    assert_type(x.atanh(), Tensor[3, 3])

    # Test angle conversions
    assert_type(x.deg2rad(), Tensor[3, 3])
    assert_type(x.rad2deg(), Tensor[3, 3])


def test_bitwise_operations() -> None:
    """Test bitwise operations preserve shape via Self."""
    x = cast(Tensor[2, 4], ...)
    y = cast(Tensor[2, 4], ...)

    assert_type(x.bitwise_and(y), Tensor[2, 4])
    assert_type(x.bitwise_or(y), Tensor[2, 4])
    assert_type(x.bitwise_xor(y), Tensor[2, 4])
    assert_type(x.bitwise_not(), Tensor[2, 4])
    assert_type(x.bitwise_left_shift(y), Tensor[2, 4])
    assert_type(x.bitwise_right_shift(y), Tensor[2, 4])


def test_validation_operations() -> None:
    """Test validation operations preserve shape via Self."""
    x = cast(Tensor[5, 5], ...)

    assert_type(x.isnan(), Tensor[5, 5])
    assert_type(x.isinf(), Tensor[5, 5])
    assert_type(x.isfinite(), Tensor[5, 5])
    assert_type(x.isreal(), Tensor[5, 5])
    assert_type(x.isposinf(), Tensor[5, 5])
    assert_type(x.isneginf(), Tensor[5, 5])


def test_minmax_operations() -> None:
    """Test min/max operations preserve shape via Self."""
    x = cast(Tensor[3, 4], ...)
    y = cast(Tensor[3, 4], ...)

    assert_type(x.maximum(y), Tensor[3, 4])
    assert_type(x.minimum(y), Tensor[3, 4])
    assert_type(x.fmax(y), Tensor[3, 4])
    assert_type(x.fmin(y), Tensor[3, 4])


def test_linalg_operations() -> None:
    """Test linear algebra operations preserve shape via Self."""
    x = cast(Tensor[4, 4], ...)

    assert_type(x.cholesky(), Tensor[4, 4])
    assert_type(x.inverse(), Tensor[4, 4])
    # NOTE: det and trace return scalars, not shaped tensors - use meta-shape
    # assert_type(x.det(), Tensor[4, 4])
    # assert_type(x.logdet(), Tensor[4, 4])
    assert_type(x.matrix_power(3), Tensor[4, 4])
    # assert_type(x.trace(), Tensor[4, 4])


def test_indexing_operations() -> None:
    """Test indexing operations preserve shape via Self."""
    x = cast(Tensor[6, 8], ...)
    mask = cast(Tensor[6, 8], ...)
    source = cast(Tensor[6, 8], ...)
    index = cast(Tensor[6, 8], ...)

    assert_type(x.masked_scatter(mask, source), Tensor[6, 8])
    assert_type(x.masked_scatter_(mask, source), Tensor[6, 8])
    assert_type(x.index_copy(0, index, source), Tensor[6, 8])
    assert_type(x.index_copy_(0, index, source), Tensor[6, 8])
    assert_type(x.index_fill(0, index, 1.0), Tensor[6, 8])
    assert_type(x.index_fill_(0, index, 1.0), Tensor[6, 8])
    # NOTE: take and take_along_dim use index shape, not Self - use meta-shape
    # assert_type(x.take(index), Tensor[6, 8])
    # assert_type(x.take_along_dim(index, 0), Tensor[6, 8])


def test_random_operations() -> None:
    """Test random operations preserve shape via Self."""
    x = cast(Tensor[7, 9], ...)

    assert_type(x.bernoulli(), Tensor[7, 9])
    assert_type(x.bernoulli_(), Tensor[7, 9])
    assert_type(x.normal_(), Tensor[7, 9])
    assert_type(x.random_(), Tensor[7, 9])
    assert_type(x.uniform_(), Tensor[7, 9])
