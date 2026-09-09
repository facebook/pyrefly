# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, cast, Literal

import numpy as np
from shape_extensions import assert_shape, IntTuple

GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_binary_ufuncs_fall_back_for_unknown_shapes",
}


def make_array(shape: Any, value: Any = 1.0) -> Any:
    return np.full(shape, value)


def test_exponential_and_log_preserve_shape() -> None:
    a = np.full(4, 2.0)
    b = np.ones((3, 4))

    assert_shape(np.exp(a).shape, (4,))
    assert_shape(np.log(a).shape, (4,))
    assert_shape(np.log2(a).shape, (4,))
    assert_shape(np.log10(a).shape, (4,))
    assert_shape(np.sqrt(b).shape, (3, 4))
    assert_shape(np.power(b, 2).shape, (3, 4))


def test_unary_ufuncs_preserve_shape() -> None:
    a = np.ones((2, 3))

    assert_shape(np.arcsin(a).shape, (2, 3))
    assert_shape(np.absolute(a).shape, (2, 3))
    assert_shape(np.exp2(a).shape, (2, 3))
    assert_shape(np.isfinite(a).shape, (2, 3))
    assert_shape(np.sin(a).shape, (2, 3))
    assert_shape(np.square(a).shape, (2, 3))
    # @lint-ignore SPELL
    assert np.arcsin.nin == 1
    assert np.abs.nout == 1
    # @lint-ignore SPELL
    assert np.absolute.nin == 1
    assert np.absolute.nout == 1
    assert np.absolute(-1.0) == 1.0


def test_binary_ufuncs_preserve_matrix_shape() -> None:
    a = np.ones((3, 4))
    b = np.full((3, 4), 2.0)

    assert_shape(np.minimum(a, b).shape, (3, 4))
    assert_shape(np.maximum(a, b).shape, (3, 4))
    assert_shape(np.arctan2(a, b).shape, (3, 4))


def test_binary_ufunc_objects_broadcast_arrays_and_scalars() -> None:
    matrix = np.ones((3, 1))
    row = np.full((1, 4), 2.0)

    assert_shape(np.add(matrix, row).shape, (3, 4))
    assert_shape(np.multiply(matrix, 2.0).shape, (3, 1))
    assert_shape(np.power(matrix, row).shape, (3, 4))
    assert_shape(np.equal(2.0, row).shape, (1, 4))
    # @lint-ignore SPELL
    assert np.add.nin == 2
    assert np.add.nout == 1
    # @lint-ignore SPELL
    assert np.arctan2.nin == 2
    # @lint-ignore SPELL
    assert np.matmul.nin == 2


def test_multi_output_ufuncs_preserve_shapes() -> None:
    matrix = np.ones((2, 3))
    row = np.full((1, 3), 2.0)

    fraction, exponent = np.frexp(matrix)
    assert_shape(fraction.shape, (2, 3))
    assert_shape(exponent.shape, (2, 3))
    fractional, integral = np.modf(matrix)
    assert_shape(fractional.shape, (2, 3))
    assert_shape(integral.shape, (2, 3))
    quotient, remainder = np.divmod(matrix, row)
    assert_shape(quotient.shape, (2, 3))
    assert_shape(remainder.shape, (2, 3))

    frexp_fraction_out = np.empty_like(matrix)
    frexp_exponent_out = np.empty_like(matrix)
    frexp_result = np.frexp(matrix, frexp_fraction_out, frexp_exponent_out)
    assert frexp_result[0] is frexp_fraction_out
    assert frexp_result[1] is frexp_exponent_out

    divmod_quotient_out = np.empty_like(matrix)
    divmod_remainder_out = np.empty_like(matrix)
    divmod_result = np.divmod(matrix, row, divmod_quotient_out, divmod_remainder_out)
    assert divmod_result[0] is divmod_quotient_out
    assert divmod_result[1] is divmod_remainder_out


def test_vector_gufuncs_track_core_and_batch_dimensions() -> None:
    matrices = cast(
        "np.ndarray[tuple[Literal[4], Literal[2], Literal[3]]]",
        make_array((4, 2, 3)),
    )
    vectors = np.ones((4, 3))
    right_matrices = cast(
        "np.ndarray[tuple[Literal[4], Literal[3], Literal[5]]]",
        make_array((4, 3, 5)),
    )

    assert_shape(np.matvec(matrices, vectors).shape, (4, 2))
    assert_shape(np.vecdot(vectors, vectors).shape, (4,))
    assert_shape(np.vecmat(vectors, right_matrices).shape, (4, 5))
    try:
        np.vecdot(  # E: core dimension 'n' has conflicting extents 3 and 4
            np.ones(3), np.ones(4)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject mismatched core dimensions")


def test_extrema_broadcast_row_vector_over_matrix() -> None:
    matrix = np.ones((3, 4))
    row = np.full(4, 2.0)

    assert_shape(np.minimum(matrix, row).shape, (3, 4))
    assert_shape(np.minimum(row, matrix).shape, (3, 4))
    assert_shape(np.maximum(matrix, row).shape, (3, 4))
    assert_shape(np.maximum(row, matrix).shape, (3, 4))


def test_binary_ufuncs_broadcast_higher_rank_arrays() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[1], Literal[5]]]",
        make_array((2, 3, 1, 5)),
    )
    b = cast(
        "np.ndarray[tuple[Literal[1], Literal[3], Literal[4], Literal[1]]]",
        make_array((1, 3, 4, 1), 2.0),
    )

    assert_shape(np.minimum(a, b).shape, (2, 3, 4, 5))
    assert_shape(np.maximum(a, b).shape, (2, 3, 4, 5))
    assert_shape(np.arctan2(a, b).shape, (2, 3, 4, 5))


def test_binary_ufuncs_fall_back_for_unknown_shapes() -> None:
    unknown = cast("np.ndarray", make_array((2, 3)))
    concrete = np.ones((2, 3), dtype=np.int64)

    minimum = np.minimum(unknown, concrete)
    # TODO(stroxler): Preserve more precision when broadcasting against unknown shapes.
    assert_type(minimum, np.ndarray[IntTuple])
    assert minimum.shape == (2, 3)


def test_binary_ufuncs_keep_broad_mixed_dtype_results() -> None:
    left = np.ones((2, 1), dtype=np.float32)
    right = np.ones((1, 3), dtype=np.int64)

    minimum = np.minimum(left, right)
    assert_type(minimum, np.ndarray[[2, 3], Any])
    assert_shape(minimum.shape, (2, 3))


def test_binary_ufuncs_reject_incompatible_broadcast() -> None:
    a = np.ones((3, 4))
    b = np.ones(5)

    assert_shape(np.minimum(a, np.ones((3, 4))).shape, (3, 4))
    try:
        np.minimum(  # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[4] with dimension Int[5] at position 1
            a, b
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject incompatible shapes")


def test_binary_ufuncs_remain_positional_only() -> None:
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    assert_shape(np.minimum(a, b).shape, (3, 4))
    try:
        np.minimum(  # E: No matching overload found
            x1=a,
            x2=b,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("expected NumPy to require positional arguments")


def test_trig_preserves_shape() -> None:
    angles = np.ones((2, 3))

    assert_shape(np.sin(angles).shape, (2, 3))
    assert_shape(np.cos(angles).shape, (2, 3))
    assert_shape(np.tan(angles).shape, (2, 3))
    assert_shape(np.arcsin(np.full((2, 3), 0.5)).shape, (2, 3))


def test_rounding_preserves_shape() -> None:
    a = np.full(5, -1.7)

    assert_shape(np.floor(a).shape, (5,))
    assert_shape(np.ceil(a).shape, (5,))
    assert_shape(np.round(a).shape, (5,))
    assert_shape(np.trunc(a).shape, (5,))
    assert_shape(np.clip(a, -1.0, 2.0).shape, (5,))
