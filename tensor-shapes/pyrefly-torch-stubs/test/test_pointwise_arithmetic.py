# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_arithmetic_function_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(torch.add(left, right).shape, (2, 3))
    assert_shape(torch.pow(left, right).shape, (2, 3))
    assert_shape(torch.atan2(left, right).shape, (2, 3))
    assert_shape(torch.hypot(left, right).shape, (2, 3))
    assert_shape(torch.lerp(left, right, 0.5).shape, (2, 3))
    assert_shape(torch.fmod(left, right).shape, (2, 3))
    assert_shape(torch.remainder(left, right).shape, (2, 3))
    assert_shape(torch.copysign(left, right).shape, (2, 3))
    assert_shape(torch.nextafter(left, right).shape, (2, 3))


def test_arithmetic_method_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(left.add(right).shape, (2, 3))
    assert_shape(left.pow(right).shape, (2, 3))
    assert_shape(left.maximum(right).shape, (2, 3))
    assert_shape(left.minimum(right).shape, (2, 3))
    assert_shape(left.fmax(right).shape, (2, 3))
    assert_shape(left.fmin(right).shape, (2, 3))
    assert_shape(torch.maximum(left, right).shape, (2, 3))
    assert_shape(torch.minimum(left, right).shape, (2, 3))
    assert_shape(torch.fmax(left, right).shape, (2, 3))
    assert_shape(torch.fmin(left, right).shape, (2, 3))
    assert_shape(left.atan2(right).shape, (2, 3))
    assert_shape(left.hypot(right).shape, (2, 3))
    assert_shape(left.lerp(right, 0.5).shape, (2, 3))
    assert_shape(left.fmod(right).shape, (2, 3))
    assert_shape(left.remainder(right).shape, (2, 3))
    assert_shape(left.copysign(right).shape, (2, 3))
    assert_shape(left.nextafter(right).shape, (2, 3))


def test_arithmetic_operator_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape((left + right).shape, (2, 3))
    assert_shape((left**right).shape, (2, 3))


def test_arithmetic_scalar_shapes() -> None:
    x = torch.ones((2, 3))

    assert_shape((x + 1.0).shape, (2, 3))
    assert_shape(torch.add(x, 1.0).shape, (2, 3))
    assert_shape(torch.sub(x, 1.0).shape, (2, 3))
    assert_shape(x.add(1.0).shape, (2, 3))
    assert_shape(x.sub(1.0).shape, (2, 3))
    assert_shape(torch.mul(x, 2.0).shape, (2, 3))
    assert_shape(x.div(2.0).shape, (2, 3))


def test_nonliteral_scalar_expression_shapes() -> None:
    n_bits = 4
    scale = 3
    offset = 2

    matrix = torch.randn((4, 1))
    assert_shape((2 * matrix / (2 ** (n_bits * 1.0) - 1.0) - 1.0).shape, (4, 1))
    assert_shape((matrix * (2 ** (scale * 1.0))).shape, (4, 1))
    assert_shape((matrix + (2 ** (offset * 1.0))).shape, (4, 1))


def test_inplace_arithmetic_preserves_shape() -> None:
    tensor = torch.ones((2, 3))
    other = torch.ones((1, 3))

    tensor += other
    tensor -= 1
    tensor *= 2
    tensor /= 2
    tensor //= 1
    tensor %= 2
    tensor **= 2
    assert_shape(tensor.shape, (2, 3))


def test_inplace_arithmetic_rejects_invalid_operands() -> None:
    incompatible = torch.ones((2, 3))
    with assert_raises(RuntimeError):
        # E: in-place operation cannot expand the receiver shape
        incompatible += torch.ones((4, 5))

    would_expand = torch.ones((1, 3))
    with assert_raises(RuntimeError):
        # E: in-place operation cannot expand the receiver shape
        would_expand += torch.ones((2, 3))

    tensor = torch.ones((2, 3))
    with assert_raises(TypeError):
        tensor += object()  # E: is not assignable to parameter `other`


def test_arithmetic_rejects_incompatible_shapes() -> None:
    left = torch.ones((2, 3))
    right = torch.ones((4, 5))
    assert_shape((left + torch.ones((2, 3))).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left + right

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.add(left, right)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left.add(right)


def test_arithmetic_rejects_invalid_scalar_options() -> None:
    tensor = torch.ones((2, 3))

    with assert_raises(TypeError):
        torch.mul(tensor, object())  # E: is not assignable to parameter `other`
    with assert_raises(TypeError):
        torch.allclose(tensor, tensor, equal_nan="yes")  # E: is not assignable


if TYPE_CHECKING:

    def check_any_operand(tensor: Tensor[[2, 5]], other: Any) -> None:
        assert_type(tensor + other, Any)
        assert_type(other + tensor, Any)

    def check_gradual_broadcast(
        concrete: Tensor[[2, 3]],
        gradual_dimension: Tensor[[int, 3]],
        gradual_shape: Tensor[IntTuple],
    ) -> None:
        assert_type(concrete + gradual_dimension, Tensor[[2, 3]])
        assert_type(gradual_dimension + gradual_dimension, Tensor[[int, 3]])
        assert_type(concrete + gradual_shape, Tensor[IntTuple])

    def check_variadic_broadcast[Left: IntTuple, Right: IntTuple](
        left: Tensor[[*Elements[Left], 3]],
        same: Tensor[[*Elements[Left], 3]],
        right: Tensor[[*Elements[Right], 3]],
        vector: Tensor[[3]],
        scalar: Tensor[[]],
    ) -> None:
        assert_type(left + same, Tensor[[*Elements[Left], 3]])
        assert_type(vector + left, Tensor[[*Elements[Left], 3]])
        assert_type(scalar + left, Tensor[[*Elements[Left], 3]])
        assert_type(left + right, Tensor[[*Elements[IntTuple], 3]])

    def check_incompatible_symbolic_broadcast[N: IntVar, M: IntVar](
        left: Tensor[[N, 3]], right: Tensor[[M, 3]]
    ) -> None:
        # E: `+` is not supported
        # E: Cannot evaluate type-level shape DSL call
        left + right

    def check_incompatible_variadic_broadcast[Batch: IntTuple](
        concrete: Tensor[[5, 10, 20]],
        variadic: Tensor[[*Elements[Batch], 20]],
    ) -> None:
        # E: Cannot evaluate type-level shape DSL call
        concrete + variadic

    def check_symbolic_arithmetic[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(left + right, Tensor[[N, M]])
        assert_type(left - right, Tensor[[N, M]])
        assert_type(left * right, Tensor[[N, M]])
        assert_type(left / right, Tensor[[N, M]])
        assert_type(left**right, Tensor[[N, M]])
        assert_type(torch.add(left, right), Tensor[[N, M]])
        assert_type(torch.sub(left, right), Tensor[[N, M]])
        assert_type(torch.mul(left, right), Tensor[[N, M]])
        assert_type(torch.div(left, right), Tensor[[N, M]])
        assert_type(torch.pow(left, right), Tensor[[N, M]])
        assert_type(left.add(right), Tensor[[N, M]])
        assert_type(left.sub(right), Tensor[[N, M]])
        assert_type(left.mul(right), Tensor[[N, M]])
        assert_type(left.div(right), Tensor[[N, M]])
        assert_type(left.pow(right), Tensor[[N, M]])
        assert_type(torch.atan2(left, right), Tensor[[N, M]])
        assert_type(left.atan2(right), Tensor[[N, M]])
        assert_type(torch.hypot(left, right), Tensor[[N, M]])
        assert_type(torch.lerp(left, right, 0.5), Tensor[[N, M]])
        assert_type(torch.fmod(left, right), Tensor[[N, M]])
        assert_type(torch.remainder(left, right), Tensor[[N, M]])
        assert_type(torch.copysign(left, right), Tensor[[N, M]])
        assert_type(torch.nextafter(left, right), Tensor[[N, M]])
        assert_type(left.hypot(right), Tensor[[N, M]])
        assert_type(left.lerp(right, 0.5), Tensor[[N, M]])
        assert_type(left.fmod(right), Tensor[[N, M]])
        assert_type(left.remainder(right), Tensor[[N, M]])
        assert_type(left.copysign(right), Tensor[[N, M]])
        assert_type(left.nextafter(right), Tensor[[N, M]])

    def check_scalar_operators[N: IntVar, M: IntVar](tensor: Tensor[[N, M]]) -> None:
        assert_type(tensor + 1, Tensor[[N, M]])
        assert_type(tensor - 1.0, Tensor[[N, M]])
        assert_type(tensor * 1j, Tensor[[N, M]])
        assert_type(tensor % 2, Tensor[[N, M]])
        assert_type(tensor / 2, Tensor[[N, M]])
        assert_type(tensor // 2, Tensor[[N, M]])
        assert_type(tensor.__iadd__(1), Tensor[[N, M]])
        assert_type(tensor.__isub__(1), Tensor[[N, M]])
        assert_type(tensor.__imul__(1), Tensor[[N, M]])
        assert_type(tensor.__itruediv__(1), Tensor[[N, M]])
        assert_type(tensor.__ifloordiv__(1), Tensor[[N, M]])
        assert_type(tensor.__imod__(1), Tensor[[N, M]])
        assert_type(tensor.__ipow__(1), Tensor[[N, M]])

    def check_incompatible_power_shapes(
        left: Tensor[[2, 3]], right: Tensor[[4, 5]]
    ) -> None:
        # E: Cannot broadcast dimension
        left**right
        # E: Cannot broadcast dimension
        torch.pow(left, right)
        # E: Cannot broadcast dimension
        left.pow(right)
