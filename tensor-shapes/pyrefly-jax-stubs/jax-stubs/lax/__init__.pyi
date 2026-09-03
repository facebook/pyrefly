# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, overload, Sequence

from jax._array import Array
from jax._shapes import (
    broadcast_to_rank_shape,
    collapse_shape,
    collapse_to_end_shape,
    concatenate_shape,
    lax_broadcast,
    lax_squeeze_shape,
    permute_shape,
    stack_shape,
)
from shape_extensions import (
    Elements,
    Flag,
    Int,
    IntTuple,
    IntTuples,
    IntVar,
    MapIntTuples,
)

from . import linalg as linalg

type _Shape = IntTuple
type _Scalar = int | float | complex
type _Axis = int | tuple[int, ...] | None

# Unary elementwise operators
def abs[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def acos[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def acosh[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def asin[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def asinh[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def atan[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def atanh[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def bessel_i0e[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def bessel_i1e[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def bitwise_not[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def cbrt[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def ceil[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def clz[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def conj[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def cos[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def cosh[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def digamma[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def erf[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def erf_inv[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def erfc[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def exp[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def exp2[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def expm1[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def floor[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def imag[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def integer_pow[Shape: _Shape](x: Array[Shape], y: int) -> Array[Shape]: ...
def is_finite[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def lgamma[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def log[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def log1p[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def logistic[Shape: _Shape](
    x: Array[Shape], *, accuracy: Any = None
) -> Array[Shape]: ...
def neg[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def population_count[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def real[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def reciprocal[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def round[Shape: _Shape](
    x: Array[Shape], rounding_method: Any = ...
) -> Array[Shape]: ...
def rsqrt[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def sign[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def sin[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def sinh[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def sqrt[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def square[Shape: _Shape](x: Array[Shape]) -> Array[Shape]: ...
def tan[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...
def tanh[Shape: _Shape](x: Array[Shape], *, accuracy: Any = None) -> Array[Shape]: ...

# Binary elementwise operators with strict rank-matching broadcasting
@overload
def add[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def add[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def add[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def atan2[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def atan2[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def atan2[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_and[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def bitwise_and[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def bitwise_and[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_or[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def bitwise_or[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def bitwise_or[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_xor[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def bitwise_xor[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def bitwise_xor[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def complex[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def complex[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def complex[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def div[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def div[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def div[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def eq[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def eq[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def eq[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def ge[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def ge[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def ge[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def gt[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def gt[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def gt[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def igamma[Shape: _Shape](a: Array[Shape], x: _Scalar, /) -> Array[Shape]: ...
@overload
def igamma[Shape: _Shape](a: _Scalar, x: Array[Shape], /) -> Array[Shape]: ...
@overload
def igamma[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1], x: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def igamma_grad_a[Shape: _Shape](a: Array[Shape], x: _Scalar, /) -> Array[Shape]: ...
@overload
def igamma_grad_a[Shape: _Shape](a: _Scalar, x: Array[Shape], /) -> Array[Shape]: ...
@overload
def igamma_grad_a[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1], x: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def igammac[Shape: _Shape](a: Array[Shape], x: _Scalar, /) -> Array[Shape]: ...
@overload
def igammac[Shape: _Shape](a: _Scalar, x: Array[Shape], /) -> Array[Shape]: ...
@overload
def igammac[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1], x: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def le[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def le[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def le[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def lt[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def lt[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def lt[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def max[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def max[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def max[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def min[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def min[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def min[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def mul[Shape: _Shape](
    x: Array[Shape],
    y: _Scalar,
    /,
    *,
    out_dtype: Any = None,
) -> Array[Shape]: ...
@overload
def mul[Shape: _Shape](
    x: _Scalar,
    y: Array[Shape],
    /,
    *,
    out_dtype: Any = None,
) -> Array[Shape]: ...
@overload
def mul[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1],
    y: Array[Shape2],
    /,
    *,
    out_dtype: Any = None,
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def mulhi[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def mulhi[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def mulhi[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def ne[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def ne[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def ne[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def nextafter[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def nextafter[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def nextafter[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def polygamma[Shape: _Shape](m: Array[Shape], x: _Scalar, /) -> Array[Shape]: ...
@overload
def polygamma[Shape: _Shape](m: _Scalar, x: Array[Shape], /) -> Array[Shape]: ...
@overload
def polygamma[Shape1: _Shape, Shape2: _Shape](
    m: Array[Shape1], x: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def pow[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def pow[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def pow[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def rem[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def rem[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def rem[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def shift_left[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def shift_left[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def shift_left[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def shift_right_arithmetic[Shape: _Shape](
    x: Array[Shape], y: _Scalar, /
) -> Array[Shape]: ...
@overload
def shift_right_arithmetic[Shape: _Shape](
    x: _Scalar, y: Array[Shape], /
) -> Array[Shape]: ...
@overload
def shift_right_arithmetic[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def shift_right_logical[Shape: _Shape](
    x: Array[Shape], y: _Scalar, /
) -> Array[Shape]: ...
@overload
def shift_right_logical[Shape: _Shape](
    x: _Scalar, y: Array[Shape], /
) -> Array[Shape]: ...
@overload
def shift_right_logical[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def sub[Shape: _Shape](x: Array[Shape], y: _Scalar, /) -> Array[Shape]: ...
@overload
def sub[Shape: _Shape](x: _Scalar, y: Array[Shape], /) -> Array[Shape]: ...
@overload
def sub[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], y: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def zeta[Shape: _Shape](x: Array[Shape], q: _Scalar, /) -> Array[Shape]: ...
@overload
def zeta[Shape: _Shape](x: _Scalar, q: Array[Shape], /) -> Array[Shape]: ...
@overload
def zeta[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1], q: Array[Shape2], /
) -> Array[lax_broadcast(Shape1, Shape2)]: ...

# -----------------------------------------------------------------------------
# Array Creation & Constants
# -----------------------------------------------------------------------------

@overload
def broadcasted_iota[Shape: _Shape](
    dtype: Any,
    shape: Shape,
    dimension: int,
    *,
    out_sharding: Any = None,
) -> Array[Shape]: ...
@overload
def broadcasted_iota(
    dtype: Any,
    shape: Sequence[int] | int,
    dimension: int,
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def empty(shape: tuple[()], dtype: Any, *, out_sharding: Any = None) -> Array[[]]: ...
@overload
def empty[N: IntVar](
    shape: Int[N], dtype: Any, *, out_sharding: Any = None
) -> Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape, dtype: Any, *, out_sharding: Any = None
) -> Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int, dtype: Any, *, out_sharding: Any = None
) -> Array[IntTuple]: ...
@overload
def full(
    shape: tuple[()],
    fill_value: Any,
    dtype: Any = None,
    *,
    sharding: Any = None,
) -> Array[[]]: ...
@overload
def full[N: IntVar](
    shape: Int[N],
    fill_value: Any,
    dtype: Any = None,
    *,
    sharding: Any = None,
) -> Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape,
    fill_value: Any,
    dtype: Any = None,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: Any = None,
    *,
    sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape](
    x: Array[Shape],
    fill_value: Any,
    dtype: Any = None,
    shape: None = None,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full_like[N: IntVar](
    x: Any,
    fill_value: Any,
    dtype: Any = None,
    shape: Int[N] = ...,
    *,
    sharding: Any = None,
) -> Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    x: Any,
    fill_value: Any,
    dtype: Any = None,
    shape: Shape = ...,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full_like(
    x: Any,
    fill_value: Any,
    dtype: Any = None,
    shape: Sequence[int] | int | None = None,
    *,
    sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def iota[N: IntVar](dtype: Any, size: Int[N]) -> Array[[N]]: ...
@overload
def iota(dtype: Any, size: int) -> Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Shape Manipulation, Slicing & Reshaping
# -----------------------------------------------------------------------------

@overload
def broadcast[Shape: _Shape](
    operand: Array[Shape],
    sizes: tuple[()],
    *,
    out_sharding: Any = None,
) -> Array[Shape]: ...
@overload
def broadcast[Shape: _Shape, D0: IntVar](
    operand: Array[Shape],
    sizes: tuple[Int[D0]],
    *,
    out_sharding: Any = None,
) -> Array[[D0, *Elements[Shape]]]: ...
@overload
def broadcast[Shape: _Shape, D0: IntVar, D1: IntVar](
    operand: Array[Shape],
    sizes: tuple[Int[D0], Int[D1]],
    *,
    out_sharding: Any = None,
) -> Array[[D0, D1, *Elements[Shape]]]: ...
@overload
def broadcast[Shape: _Shape, D0: IntVar, D1: IntVar, D2: IntVar](
    operand: Array[Shape],
    sizes: tuple[Int[D0], Int[D1], Int[D2]],
    *,
    out_sharding: Any = None,
) -> Array[[D0, D1, D2, *Elements[Shape]]]: ...
@overload
def broadcast(
    operand: Any,
    sizes: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def broadcast_in_dim[Shape: _Shape](
    operand: Any,
    shape: Shape,
    broadcast_dimensions: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[Shape]: ...
@overload
def broadcast_in_dim(
    operand: Any,
    shape: Sequence[int] | int,
    broadcast_dimensions: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def broadcast_like[Shape: _Shape](
    arr: Any,
    like_arr: Array[Shape],
) -> Array[Shape]: ...
@overload
def broadcast_like(
    arr: Any,
    like_arr: Any,
) -> Array[IntTuple]: ...
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
@overload
def broadcast_to_rank[Shape: _Shape, Rank: Flag[int]](
    x: Array[Shape],
    rank: Rank,
) -> Array[broadcast_to_rank_shape(Shape, Rank)]: ...
@overload
def broadcast_to_rank(
    x: Any,
    rank: int,
) -> Array[IntTuple]: ...
@overload
def collapse[Shape: _Shape, Start: Flag[int]](
    operand: Array[Shape],
    start_dimension: Start,
    stop_dimension: None = None,
) -> Array[collapse_to_end_shape(Shape, Start)]: ...
@overload
def collapse[Shape: _Shape, Start: Flag[int], Stop: Flag[int]](
    operand: Array[Shape],
    start_dimension: Start,
    stop_dimension: Stop,
) -> Array[collapse_shape(Shape, Start, Stop)]: ...
@overload
def collapse(
    operand: Any,
    start_dimension: int,
    stop_dimension: int | None = None,
) -> Array[IntTuple]: ...
@overload
def concatenate[Shapes: IntTuples, Dimension: Flag[int] = 0](
    operands: MapIntTuples[lambda S: Array[S], Shapes],
    dimension: Dimension = 0,
) -> Array[concatenate_shape(Shapes, Dimension)]: ...
@overload
def concatenate(
    operands: Any,
    dimension: int = 0,
) -> Array[IntTuple]: ...
def expand_dims(
    array: Any,
    dimensions: Sequence[int],
) -> Array[IntTuple]: ...
def pad(
    operand: Any,
    padding_value: Any,
    padding_config: Sequence[tuple[int, int, int]],
) -> Array[IntTuple]: ...
def padtype_to_pads(
    in_shape: Sequence[int],
    window_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str,
) -> list[tuple[int, int]]: ...
@overload
def reshape[NewShape: _Shape](
    operand: Any,
    new_sizes: NewShape,
    dimensions: Sequence[int] | None = None,
    *,
    out_sharding: Any = None,
) -> Array[NewShape]: ...
@overload
def reshape(
    operand: Any,
    new_sizes: Sequence[int] | int,
    dimensions: Sequence[int] | None = None,
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def rev[Shape: _Shape](
    operand: Array[Shape],
    dimensions: Sequence[int],
) -> Array[Shape]: ...
def slice(
    operand: Any,
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None = None,
) -> Array[IntTuple]: ...
def slice_in_dim(
    operand: Any,
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    axis: int = 0,
) -> Array[IntTuple]: ...
def split(
    operand: Any,
    sizes: Sequence[int],
    axis: int = 0,
) -> list[Array[IntTuple]]: ...
@overload
def squeeze[Shape: _Shape, Dims: Flag[tuple[int, ...]]](
    array: Array[Shape],
    dimensions: Dims,
) -> Array[lax_squeeze_shape(Shape, Dims)]: ...
@overload
def squeeze(
    array: Any,
    dimensions: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def stack[Shapes: IntTuples, Axis: Flag[int] = 0](
    operands: MapIntTuples[lambda S: Array[S], Shapes],
    axis: Axis = 0,
) -> Array[stack_shape(Shapes, Axis)]: ...
@overload
def stack(
    operands: Any,
    axis: int = 0,
) -> Array[IntTuple]: ...
def tile(
    operand: Any,
    reps: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def transpose[Shape: _Shape, Permutation: Flag[_Axis]](
    operand: Array[Shape],
    permutation: Permutation,
) -> Array[permute_shape(Shape, Permutation)]: ...
@overload
def transpose(
    operand: Any,
    permutation: Sequence[int],
) -> Array[IntTuple]: ...
def unstack(
    x: Any,
    axis: int = 0,
) -> tuple[Array[IntTuple], ...]: ...

# Data types & bitcasting
def bitcast_convert_type(
    operand: Any,
    new_dtype: Any,
) -> Array[IntTuple]: ...
@overload
def convert_element_type[Shape: _Shape](
    operand: Array[Shape],
    new_dtype: Any,
) -> Array[Shape]: ...
@overload
def convert_element_type(
    operand: _Scalar | bool,
    new_dtype: Any,
) -> Array[[]]: ...
@overload
def convert_element_type(
    operand: Any,
    new_dtype: Any,
) -> Array[IntTuple]: ...
