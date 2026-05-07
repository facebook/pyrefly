# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 2 Extended: Additional operations smoke tests
# Tests for mathematical, bitwise, comparison, and activation operations
from typing import assert_type

import torch
from torch import Tensor

# ==== Additional Mathematical Operations (Binary) ====


# Test: atan2 - element-wise arctangent
def test_atan2():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.atan2(a, b)
    assert_type(result, Tensor[2, 3])


def test_atan2_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = a.atan2(b)
    assert_type(result, Tensor[2, 3])


# Test: hypot - element-wise hypotenuse
def test_hypot():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.hypot(a, b)
    assert_type(result, Tensor[2, 3])


def test_hypot_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[3, 4] = torch.randn(3, 4)
    result = a.hypot(b)
    assert_type(result, Tensor[3, 4])


# Test: lerp - linear interpolation
def test_lerp():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.lerp(a, b, 0.5)
    assert_type(result, Tensor[2, 3])


def test_lerp_method():
    a: Tensor[4, 5] = torch.randn(4, 5)
    b: Tensor[4, 5] = torch.randn(4, 5)
    result = a.lerp(b, 0.3)
    assert_type(result, Tensor[4, 5])


# Test: fmod - element-wise modulo
def test_fmod():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.fmod(a, b)
    assert_type(result, Tensor[2, 3])


def test_fmod_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[3, 4] = torch.randn(3, 4)
    result = a.fmod(b)
    assert_type(result, Tensor[3, 4])


# Test: remainder - element-wise remainder
def test_remainder():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.remainder(a, b)
    assert_type(result, Tensor[2, 3])


def test_remainder_method():
    a: Tensor[2, 2] = torch.randn(2, 2)
    b: Tensor[2, 2] = torch.randn(2, 2)
    result = a.remainder(b)
    assert_type(result, Tensor[2, 2])


# Test: copysign
def test_copysign():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.copysign(a, b)
    assert_type(result, Tensor[2, 3])


def test_copysign_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    b: Tensor[3, 3] = torch.randn(3, 3)
    result = a.copysign(b)
    assert_type(result, Tensor[3, 3])


# Test: nextafter
def test_nextafter():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.nextafter(a, b)
    assert_type(result, Tensor[2, 3])


def test_nextafter_method():
    a: Tensor[2, 4] = torch.randn(2, 4)
    b: Tensor[2, 4] = torch.randn(2, 4)
    result = a.nextafter(b)
    assert_type(result, Tensor[2, 4])


# ==== Additional Mathematical Operations (Unary) ====


# Test: erf - error function
def test_erf():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.erf(a)
    assert_type(result, Tensor[2, 3])


def test_erf_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    result = a.erf()
    assert_type(result, Tensor[3, 4])


# Test: erfc - complementary error function
def test_erfc():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.erfc(a)
    assert_type(result, Tensor[2, 3])


def test_erfc_method():
    a: Tensor[4, 5] = torch.randn(4, 5)
    result = a.erfc()
    assert_type(result, Tensor[4, 5])


# Test: erfinv - inverse error function
def test_erfinv():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.erfinv(a)
    assert_type(result, Tensor[2, 3])


def test_erfinv_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    result = a.erfinv()
    assert_type(result, Tensor[3, 3])


# Test: lgamma - log gamma function
def test_lgamma():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.lgamma(a)
    assert_type(result, Tensor[2, 3])


def test_lgamma_method():
    a: Tensor[2, 4] = torch.randn(2, 4)
    result = a.lgamma()
    assert_type(result, Tensor[2, 4])


# Test: digamma
def test_digamma():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.digamma(a)
    assert_type(result, Tensor[2, 3])


def test_digamma_method():
    a: Tensor[3, 5] = torch.randn(3, 5)
    result = a.digamma()
    assert_type(result, Tensor[3, 5])


# Test: polygamma
def test_polygamma():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.polygamma(1, a)
    assert_type(result, Tensor[2, 3])


def test_polygamma_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    result = a.polygamma(2)
    assert_type(result, Tensor[3, 4])


# Test: asinh - inverse hyperbolic sine
def test_asinh():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.asinh(a)
    assert_type(result, Tensor[2, 3])


def test_asinh_method():
    a: Tensor[4, 4] = torch.randn(4, 4)
    result = a.asinh()
    assert_type(result, Tensor[4, 4])


# Test: acosh - inverse hyperbolic cosine
def test_acosh():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.acosh(a)
    assert_type(result, Tensor[2, 3])


def test_acosh_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    result = a.acosh()
    assert_type(result, Tensor[3, 3])


# Test: atanh - inverse hyperbolic tangent
def test_atanh():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.atanh(a)
    assert_type(result, Tensor[2, 3])


def test_atanh_method():
    a: Tensor[2, 5] = torch.randn(2, 5)
    result = a.atanh()
    assert_type(result, Tensor[2, 5])


# Test: deg2rad - degrees to radians
def test_deg2rad():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.deg2rad(a)
    assert_type(result, Tensor[2, 3])


def test_deg2rad_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    result = a.deg2rad()
    assert_type(result, Tensor[3, 4])


# Test: rad2deg - radians to degrees
def test_rad2deg():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.rad2deg(a)
    assert_type(result, Tensor[2, 3])


def test_rad2deg_method():
    a: Tensor[4, 2] = torch.randn(4, 2)
    result = a.rad2deg()
    assert_type(result, Tensor[4, 2])


# ==== Bitwise Operations ====


# Test: bitwise_and
def test_bitwise_and():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_and(a, b)
    assert_type(result, Tensor[2, 3])


def test_bitwise_and_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    b: Tensor[3, 3] = torch.randn(3, 3)
    result = a.bitwise_and(b)
    assert_type(result, Tensor[3, 3])


# Test: bitwise_or
def test_bitwise_or():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_or(a, b)
    assert_type(result, Tensor[2, 3])


def test_bitwise_or_method():
    a: Tensor[2, 4] = torch.randn(2, 4)
    b: Tensor[2, 4] = torch.randn(2, 4)
    result = a.bitwise_or(b)
    assert_type(result, Tensor[2, 4])


# Test: bitwise_xor
def test_bitwise_xor():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_xor(a, b)
    assert_type(result, Tensor[2, 3])


def test_bitwise_xor_method():
    a: Tensor[4, 4] = torch.randn(4, 4)
    b: Tensor[4, 4] = torch.randn(4, 4)
    result = a.bitwise_xor(b)
    assert_type(result, Tensor[4, 4])


# Test: bitwise_not
def test_bitwise_not():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_not(a)
    assert_type(result, Tensor[2, 3])


def test_bitwise_not_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    result = a.bitwise_not()
    assert_type(result, Tensor[3, 4])


# Test: bitwise_left_shift
def test_bitwise_left_shift():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_left_shift(a, b)
    assert_type(result, Tensor[2, 3])


def test_bitwise_left_shift_method():
    a: Tensor[2, 2] = torch.randn(2, 2)
    b: Tensor[2, 2] = torch.randn(2, 2)
    result = a.bitwise_left_shift(b)
    assert_type(result, Tensor[2, 2])


# Test: bitwise_right_shift
def test_bitwise_right_shift():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.bitwise_right_shift(a, b)
    assert_type(result, Tensor[2, 3])


def test_bitwise_right_shift_method():
    a: Tensor[3, 5] = torch.randn(3, 5)
    b: Tensor[3, 5] = torch.randn(3, 5)
    result = a.bitwise_right_shift(b)
    assert_type(result, Tensor[3, 5])


# ==== Additional Comparison/Validation Operations ====


# Test: isclose
def test_isclose():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.isclose(a, b)
    assert_type(result, Tensor[2, 3])


def test_isclose_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[3, 4] = torch.randn(3, 4)
    result = a.isclose(b)
    assert_type(result, Tensor[3, 4])


# Test: isreal
def test_isreal():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.isreal(a)
    assert_type(result, Tensor[2, 3])


def test_isreal_method():
    a: Tensor[4, 5] = torch.randn(4, 5)
    result = a.isreal()
    assert_type(result, Tensor[4, 5])


# Test: isposinf
def test_isposinf():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.isposinf(a)
    assert_type(result, Tensor[2, 3])


def test_isposinf_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    result = a.isposinf()
    assert_type(result, Tensor[3, 3])


# Test: isneginf
def test_isneginf():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.isneginf(a)
    assert_type(result, Tensor[2, 3])


def test_isneginf_method():
    a: Tensor[2, 4] = torch.randn(2, 4)
    result = a.isneginf()
    assert_type(result, Tensor[2, 4])


# Test: maximum
def test_maximum():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.maximum(a, b)
    assert_type(result, Tensor[2, 3])


def test_maximum_method():
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[3, 4] = torch.randn(3, 4)
    result = a.maximum(b)
    assert_type(result, Tensor[3, 4])


# Test: minimum
def test_minimum():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.minimum(a, b)
    assert_type(result, Tensor[2, 3])


def test_minimum_method():
    a: Tensor[4, 2] = torch.randn(4, 2)
    b: Tensor[4, 2] = torch.randn(4, 2)
    result = a.minimum(b)
    assert_type(result, Tensor[4, 2])


# Test: fmax
def test_fmax():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.fmax(a, b)
    assert_type(result, Tensor[2, 3])


def test_fmax_method():
    a: Tensor[2, 5] = torch.randn(2, 5)
    b: Tensor[2, 5] = torch.randn(2, 5)
    result = a.fmax(b)
    assert_type(result, Tensor[2, 5])


# Test: fmin
def test_fmin():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.fmin(a, b)
    assert_type(result, Tensor[2, 3])


def test_fmin_method():
    a: Tensor[3, 3] = torch.randn(3, 3)
    b: Tensor[3, 3] = torch.randn(3, 3)
    result = a.fmin(b)
    assert_type(result, Tensor[3, 3])


# ==== Shape Preservation Tests (Various Dimensions) ====


# Test: 1D tensor operations
def test_operations_1d():
    a: Tensor[10] = torch.randn(10)
    b: Tensor[10] = torch.randn(10)

    # Binary math
    atan2_result = torch.atan2(a, b)
    assert_type(atan2_result, Tensor[10])

    # Unary math
    erf_result = torch.erf(a)
    assert_type(erf_result, Tensor[10])

    # Bitwise
    bitwise_and_result = torch.bitwise_and(a, b)
    assert_type(bitwise_and_result, Tensor[10])

    # Comparison
    isclose_result = torch.isclose(a, b)
    assert_type(isclose_result, Tensor[10])


# Test: 3D tensor operations
def test_operations_3d():
    a: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    b: Tensor[2, 3, 4] = torch.randn(2, 3, 4)

    # Binary math
    hypot_result = torch.hypot(a, b)
    assert_type(hypot_result, Tensor[2, 3, 4])

    # Unary math
    lgamma_result = torch.lgamma(a)
    assert_type(lgamma_result, Tensor[2, 3, 4])

    # Comparison
    maximum_result = torch.maximum(a, b)
    assert_type(maximum_result, Tensor[2, 3, 4])


# Test: 4D tensor operations
def test_operations_4d():
    a: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    b: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)

    # Binary math
    fmod_result = torch.fmod(a, b)
    assert_type(fmod_result, Tensor[2, 3, 4, 5])

    # Unary math
    asinh_result = torch.asinh(a)
    assert_type(asinh_result, Tensor[2, 3, 4, 5])

    # Bitwise
    bitwise_xor_result = torch.bitwise_xor(a, b)
    assert_type(bitwise_xor_result, Tensor[2, 3, 4, 5])
