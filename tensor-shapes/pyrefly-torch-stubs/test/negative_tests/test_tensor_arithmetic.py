# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test tensor arithmetic shape behavior, including scalar preservation and broadcasting."""

from typing import Any, assert_type, TYPE_CHECKING

from shape_extensions import Elements, IntTuple, IntVar

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Scalar Arithmetic - Shape Preserved
# ============================================================================


def add_scalar(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Adding scalar preserves shape (returns Self)"""
    return x + 1.0


def sub_scalar(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Subtracting scalar preserves shape"""
    return x - 1.0


def mul_scalar(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Multiplying by scalar preserves shape"""
    return x * 2.0


def div_scalar(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Dividing by scalar preserves shape"""
    return x / 2.0


def pow_scalar(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Power with scalar preserves shape"""
    return x**2


# ============================================================================
# Tensor-Tensor Arithmetic - Same Shape
# ============================================================================


def add_same_shape(x: Tensor[[2, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Adding tensors of same shape preserves shape"""
    return x + y


def sub_same_shape(x: Tensor[[2, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Subtracting tensors of same shape"""
    return x - y


def mul_same_shape(x: Tensor[[2, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Element-wise multiplication of same shape tensors"""
    return x * y


def div_same_shape(x: Tensor[[2, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Element-wise division of same shape tensors"""
    return x / y


# ============================================================================
# Tensor-Tensor Arithmetic - Broadcasting
# ============================================================================


def broadcast_1_to_n(x: Tensor[[1, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Broadcasting: dimension 1 broadcasts to any size"""
    return x + y


def broadcast_rank_extension(x: Tensor[[3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Broadcasting: lower rank tensor gets leading dimensions added"""
    return x + y


def broadcast_both_directions(x: Tensor[[2, 1]], y: Tensor[[1, 3]]) -> Tensor[[2, 3]]:
    """Broadcasting: both tensors broadcast in different dimensions"""
    return x * y


def broadcast_3d(x: Tensor[[1, 4, 1]], y: Tensor[[2, 1, 3]]) -> Tensor[[2, 4, 3]]:
    """Broadcasting: 3D tensors"""
    return x + y


def broadcast_scalar_tensor(x: Tensor[[2, 3]], y: Tensor[[]]) -> Tensor[[2, 3]]:
    """Broadcasting: scalar tensor broadcasts to any shape"""
    return x * y


# ============================================================================
# Chained Operations - Shape Still Preserved
# ============================================================================


def chained_ops(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Multiple operations in chain preserve shape"""
    return (x + 1.0) * 2.0 - 0.5


def chained_with_tensor(x: Tensor[[2, 3]], y: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Chained operations with tensor and scalars"""
    return (x + y) * 2.0 + 1.0


# ============================================================================
# Symbolic Dimensions - Shape Preserved
# ============================================================================


def add_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> Tensor[[N, M]]:
    """Scalar add with symbolic dimensions"""
    return x + 1.0


def mul_symbolic[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], y: Tensor[[N, M]]
) -> Tensor[[N, M]]:
    """Tensor multiply with symbolic dimensions"""
    return x * y


def chained_symbolic[B: IntVar, N: IntVar, M: IntVar](
    x: Tensor[[B, N, M]],
) -> Tensor[[B, N, M]]:
    """Chained ops with 3D symbolic tensor"""
    return (x + 1.0) * 2.0 / 3.0


# ============================================================================
# Wrong Return Types - Errors
# ============================================================================


def add_wrong_shape(x: Tensor[[2, 3]]) -> Tensor[[4, 5]]:
    """Arithmetic preserves shape, so a different return shape is rejected."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[4, 5]]`
    return x + 1.0


def mul_wrong_rank(x: Tensor[[2, 3]]) -> Tensor[[2, 3, 4]]:
    """Scalar multiplication preserves rank."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[2, 3, 4]]`
    return x * 2.0


# ============================================================================
# Broadcasting Errors - Incompatible Shapes
# ============================================================================


def broadcast_wrong_return(x: Tensor[[1, 3]], y: Tensor[[2, 3]]) -> Tensor[[1, 3]]:
    """Broadcast result is [2, 3], not [1, 3]."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[1, 3]]`
    return x + y


def broadcast_incompatible_dims(x: Tensor[[2, 3]], y: Tensor[[4, 5]]) -> Tensor[[4, 5]]:
    """Incompatible dimensions cannot broadcast."""
    # E: Cannot broadcast tensor shapes:
    #    Cannot broadcast dimension Int[3] with dimension Int[5] at position 1
    return x + y


# ============================================================================
# Broadcasting with Any Dimensions
# ============================================================================


def broadcast_any_dim(x: Tensor[[2, 3]], y: Tensor[[Any, 3]]) -> None:
    """Any dim is compatible with anything; prefer the non-Any side"""
    assert_type(x + y, Tensor[[2, 3]])


def broadcast_both_any(x: Tensor[[Any, 3]], y: Tensor[[Any, 3]]) -> None:
    """Any vs Any produces Any"""
    assert_type(x + y, Tensor[[Any, 3]])


# ============================================================================
# Broadcasting with Symbolic Dimensions
# ============================================================================


def broadcast_same_symbolic[N: IntVar, M: IntVar](
    x: Tensor[[N, M]], y: Tensor[[N, M]]
) -> None:
    """Same symbolic dims are compatible"""
    assert_type(x + y, Tensor[[N, M]])


def broadcast_symbolic_with_1[N: IntVar](x: Tensor[[N, 3]], y: Tensor[[1, 3]]) -> None:
    """Size(1) broadcasts to symbolic dim"""
    assert_type(x + y, Tensor[[N, 3]])


def broadcast_1_with_symbolic[N: IntVar](x: Tensor[[1, 3]], y: Tensor[[N, 3]]) -> None:
    """Symmetric: Size(1) on the left broadcasts to symbolic on the right"""
    assert_type(x + y, Tensor[[N, 3]])


def broadcast_different_symbolic[N: IntVar, M: IntVar](
    x: Tensor[[N, 3]], y: Tensor[[M, 3]]
) -> Tensor[[N, 3]]:
    """Different symbolic dimensions are not compatible for broadcasting."""
    # E: Cannot broadcast tensor shapes:
    #    Cannot broadcast dimension Int[N] with dimension Int[M] at position 0
    return x + y


# ============================================================================
# Broadcasting with Shapeless Tensors
# ============================================================================


def broadcast_shaped_with_shapeless(x: Tensor[[2, 3]], y: Tensor) -> None:
    """Shaped + shapeless = shapeless (unknown rank on shapeless side)"""
    assert_type(x + y, Tensor)


def broadcast_scalar_with_shapeless(x: Tensor[[]], y: Tensor) -> None:
    """Scalar + shapeless = shapeless"""
    assert_type(x + y, Tensor)


# ============================================================================
# Broadcasting Concrete + Unpacked (suffix matching)
# ============================================================================


def broadcast_concrete_suffix_match[Ts: IntTuple](
    x: Tensor[[3]], y: Tensor[[*Elements[Ts], 3]]
) -> None:
    """Concrete consumed by suffix → preserves prefix + middle"""
    assert_type(x + y, Tensor[[*Elements[Ts], 3]])


def broadcast_scalar_with_unpacked[Ts: IntTuple](
    x: Tensor[[]], y: Tensor[[*Elements[Ts], 3]]
) -> None:
    """Scalar + unpacked = unpacked (scalar broadcasts to anything)"""
    assert_type(x + y, Tensor[[*Elements[Ts], 3]])


def broadcast_concrete_exceeds_suffix[Ts: IntTuple](
    x: Tensor[[5, 10, 20]], y: Tensor[[*Elements[Ts], 20]]
) -> Tensor[[5, 10, 20]]:
    """Leftover concrete dims cannot align with the IntTuple middle."""
    # E: Cannot broadcast tensor shapes:
    #    Cannot broadcast concrete dims with variadic shape
    return x + y


# ============================================================================
# Broadcasting Unpacked + Unpacked (same IntTuple)
# ============================================================================


def broadcast_same_int_tuple[Ts: IntTuple](
    x: Tensor[[*Elements[Ts], 3]], y: Tensor[[*Elements[Ts], 3]]
) -> None:
    """Same IntTuple, same suffix → cancel middles, result preserves shape"""
    assert_type(x + y, Tensor[[*Elements[Ts], 3]])


def broadcast_same_int_tuple_prefix[Ts: IntTuple](
    x: Tensor[[5, *Elements[Ts]]], y: Tensor[[1, *Elements[Ts]]]
) -> None:
    """Same IntTuple, broadcast prefixes (1 broadcasts to 5)"""
    assert_type(x + y, Tensor[[5, *Elements[Ts]]])


def broadcast_same_int_tuple_prefix_extension[Ts: IntTuple](
    x: Tensor[[5, 6, *Elements[Ts]]], y: Tensor[[6, *Elements[Ts]]]
) -> None:
    """Same IntTuple, left prefix extends right (right padded with implicit 1)"""
    assert_type(x + y, Tensor[[5, 6, *Elements[Ts]]])


# ============================================================================
# Broadcasting Unpacked + Unpacked with Different IntTuples
# ============================================================================


def broadcast_different_int_tuple[Ts: IntTuple, Us: IntTuple](
    x: Tensor[[*Elements[Ts], 3]], y: Tensor[[*Elements[Us], 3]]
) -> Tensor[[*Elements[Ts], 3]]:
    """Different IntTuples degrade to shapeless batch dims."""
    # E: Returned type `Tensor[[*tuple[int, ...], 3]]` is not assignable
    #    to declared return type `Tensor[[*Elements[Ts], 3]]`
    return x + y


def broadcast_different_int_tuple_any_batch[Ts: IntTuple, Us: IntTuple](
    x: Tensor[[*Elements[Ts], 3]], y: Tensor[[*Elements[Us], 3]]
) -> Tensor[[*Elements[IntTuple], 3]]:
    """Different IntTuples are accepted with unbounded batch dims."""
    return x + y
