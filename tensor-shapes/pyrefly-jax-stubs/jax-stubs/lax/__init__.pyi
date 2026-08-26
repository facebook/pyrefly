# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Hashable, overload, Sequence

import numpy as np
from jax._array import Array
from jax._shapes import (
    broadcast_to_rank_shape,
    collapse_shape,
    collapse_to_end_shape,
    concatenate_shape,
    dot_shape,
    lax_associative_scan_shape,
    lax_axis_reduce_shape,
    lax_broadcast,
    lax_clamp_max_scalar_shape,
    lax_clamp_min_scalar_shape,
    lax_clamp_shape,
    lax_dynamic_index_in_dim_shape,
    lax_dynamic_slice_in_dim_shape,
    lax_dynamic_slice_shape,
    lax_reduce_shape,
    lax_scan_shape,
    lax_select_n_shape,
    lax_select_scalar_pred_shape,
    lax_select_shape,
    lax_sort_key_val_shape,
    lax_sort_shape,
    lax_squeeze_shape,
    permute_shape,
    stack_shape,
    top_k_shape,
)
from jax._src.lax.convolution import (
    ConvDimensionNumbers as ConvDimensionNumbers,
    ConvGeneralDilatedDimensionNumbers as ConvGeneralDilatedDimensionNumbers,
)
from jax._src.lax.fft import FftType as FftType
from jax._src.lax.lax import (
    AccuracyMode as AccuracyMode,
    DotAlgorithm as DotAlgorithm,
    DotAlgorithmPreset as DotAlgorithmPreset,
    DotDimensionNumbers as DotDimensionNumbers,
    Precision as Precision,
    PrecisionLike as PrecisionLike,
    RaggedDotDimensionNumbers as RaggedDotDimensionNumbers,
    RandomAlgorithm as RandomAlgorithm,
    RoundingMethod as RoundingMethod,
    Tolerance as Tolerance,
)
from jax._src.lax.slicing import (
    GatherDimensionNumbers as GatherDimensionNumbers,
    GatherScatterMode as GatherScatterMode,
    ScatterDimensionNumbers as ScatterDimensionNumbers,
)
from jax.typing import DTypeLike
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
    out_dtype: DTypeLike | None = None,
) -> Array[Shape]: ...
@overload
def mul[Shape: _Shape](
    x: _Scalar,
    y: Array[Shape],
    /,
    *,
    out_dtype: DTypeLike | None = None,
) -> Array[Shape]: ...
@overload
def mul[Shape1: _Shape, Shape2: _Shape](
    x: Array[Shape1],
    y: Array[Shape2],
    /,
    *,
    out_dtype: DTypeLike | None = None,
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
@overload
def betainc[Shape: _Shape](
    a: Array[Shape],
    b: Array[Shape] | _Scalar,
    x: Array[Shape] | _Scalar,
    /,
) -> Array[Shape]: ...
@overload
def betainc[Shape: _Shape](
    a: _Scalar,
    b: Array[Shape],
    x: Array[Shape] | _Scalar,
    /,
) -> Array[Shape]: ...
@overload
def betainc[Shape: _Shape](
    a: _Scalar,
    b: _Scalar,
    x: Array[Shape],
    /,
) -> Array[Shape]: ...
@overload
def betainc(
    a: _Scalar,
    b: _Scalar,
    x: _Scalar,
    /,
) -> Array[[]]: ...
@overload
def betainc(
    a: Any,
    b: Any,
    x: Any,
    /,
) -> Array[IntTuple]: ...
@overload
def fft[Shape: _Shape](
    x: Array[Shape],
    fft_type: FftType | str,
    fft_lengths: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def fft(
    x: Any,
    fft_type: FftType | str,
    fft_lengths: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def random_gamma_grad[Shape: _Shape](
    a: Array[Shape],
    x: Array[Shape],
    /,
) -> Array[Shape]: ...
@overload
def random_gamma_grad[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    x: Array[Shape2],
    /,
) -> Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def random_gamma_grad(
    a: Any,
    x: Any,
    /,
) -> Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Array Creation & Constants
# -----------------------------------------------------------------------------

@overload
def broadcasted_iota[Shape: _Shape](
    dtype: DTypeLike,
    shape: Shape,
    dimension: int,
    *,
    out_sharding: Any = None,
) -> Array[Shape]: ...
@overload
def broadcasted_iota(
    dtype: DTypeLike,
    shape: Sequence[int] | int,
    dimension: int,
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def empty(
    shape: tuple[()], dtype: DTypeLike, *, out_sharding: Any = None
) -> Array[[]]: ...
@overload
def empty[N: IntVar](
    shape: Int[N], dtype: DTypeLike, *, out_sharding: Any = None
) -> Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape, dtype: DTypeLike, *, out_sharding: Any = None
) -> Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int, dtype: DTypeLike, *, out_sharding: Any = None
) -> Array[IntTuple]: ...
@overload
def full(
    shape: tuple[()],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> Array[[]]: ...
@overload
def full[N: IntVar](
    shape: Int[N],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape](
    x: Array[Shape],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: None = None,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full_like[N: IntVar](
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Int[N] = ...,
    *,
    sharding: Any = None,
) -> Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Shape = ...,
    *,
    sharding: Any = None,
) -> Array[Shape]: ...
@overload
def full_like(
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Sequence[int] | int | None = None,
    *,
    sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def iota[N: IntVar](dtype: DTypeLike, size: Int[N]) -> Array[[N]]: ...
@overload
def iota(dtype: DTypeLike, size: int) -> Array[IntTuple]: ...

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
def unstack[Batch: IntTuple, M: IntVar](
    x: Array[[*Elements[Batch], M]],
    axis: int = 0,
) -> tuple[Array[IntTuple], ...]: ...

# Dynamic Slicing & Gather/Scatter

@overload
def dynamic_index_in_dim[
    Shape: _Shape,
    Axis: Flag[int] = 0,
    KeepDims: Flag[bool] = True,
](
    operand: Array[Shape],
    index: Any,
    axis: Axis = 0,
    keepdims: KeepDims = True,
    *,
    allow_negative_indices: bool = True,
) -> Array[lax_dynamic_index_in_dim_shape(Shape, Axis, KeepDims)]: ...
@overload
def dynamic_index_in_dim(
    operand: Any,
    index: Any,
    axis: int = 0,
    keepdims: bool = True,
    *,
    allow_negative_indices: bool = True,
) -> Array[IntTuple]: ...
@overload
def dynamic_slice[Shape: _Shape, SliceSizes: _Shape](
    operand: Array[Shape],
    start_indices: Any,
    slice_sizes: SliceSizes,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array[lax_dynamic_slice_shape(Shape, SliceSizes)]: ...
@overload
def dynamic_slice(
    operand: Any,
    start_indices: Any,
    slice_sizes: Sequence[int],
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array[IntTuple]: ...
@overload
def dynamic_slice_in_dim[Shape: _Shape, SliceSize: Flag[int], Axis: Flag[int] = 0](
    operand: Array[Shape],
    start_index: Any,
    slice_size: SliceSize,
    axis: Axis = 0,
    *,
    allow_negative_indices: bool = True,
) -> Array[lax_dynamic_slice_in_dim_shape(Shape, SliceSize, Axis)]: ...
@overload
def dynamic_slice_in_dim(
    operand: Any,
    start_index: Any,
    slice_size: int,
    axis: int = 0,
    *,
    allow_negative_indices: bool = True,
) -> Array[IntTuple]: ...
@overload
def dynamic_update_index_in_dim[Shape: _Shape](
    operand: Array[Shape],
    update: Any,
    index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array[Shape]: ...
@overload
def dynamic_update_index_in_dim(
    operand: Any,
    update: Any,
    index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array[IntTuple]: ...
@overload
def dynamic_update_slice[Shape: _Shape](
    operand: Array[Shape],
    update: Any,
    start_indices: Any,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array[Shape]: ...
@overload
def dynamic_update_slice(
    operand: Any,
    update: Any,
    start_indices: Any,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array[IntTuple]: ...
@overload
def dynamic_update_slice_in_dim[Shape: _Shape](
    operand: Array[Shape],
    update: Any,
    start_index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array[Shape]: ...
@overload
def dynamic_update_slice_in_dim(
    operand: Any,
    update: Any,
    start_index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array[IntTuple]: ...
def gather(
    operand: Array[Any],
    start_indices: Array[Any],
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Sequence[int],
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: str | GatherScatterMode | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def index_in_dim[Shape: _Shape, Axis: Flag[int] = 0, KeepDims: Flag[bool] = True](
    operand: Array[Shape],
    index: int,
    axis: Axis = 0,
    keepdims: KeepDims = True,
) -> Array[lax_dynamic_index_in_dim_shape(Shape, Axis, KeepDims)]: ...
@overload
def index_in_dim(
    operand: Any,
    index: int,
    axis: int = 0,
    keepdims: bool = True,
) -> Array[IntTuple]: ...
def index_take(
    src: Array[Any],
    idxs: Array[Any],
    axes: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def scatter[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_add[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_add(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_apply[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    func: Callable[[Any], Any],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    update_shape: Sequence[int] = (),
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_apply(
    operand: Any,
    scatter_indices: Any,
    func: Callable[[Any], Any],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    update_shape: Sequence[int] = (),
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_max[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_max(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_min[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_min(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_mul[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_mul(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...
@overload
def scatter_sub[Shape: _Shape](
    operand: Array[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[Shape]: ...
@overload
def scatter_sub(
    operand: Any,
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array[IntTuple]: ...

# Linear Algebra, Contractions & Convolutions

@overload
def batch_matmul[Batch: IntTuple, M: IntVar, K: IntVar, N: IntVar](
    lhs: Array[[*Elements[Batch], M, K]],
    rhs: Array[[*Elements[Batch], K, N]],
    precision: PrecisionLike = None,
) -> Array[[*Elements[Batch], M, N]]: ...
@overload
def batch_matmul(
    lhs: Any,
    rhs: Any,
    precision: PrecisionLike = None,
) -> Array[IntTuple]: ...
def conv(
    lhs: Any,
    rhs: Any,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...
def conv_dimension_numbers(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    dimension_numbers: Any,
) -> ConvDimensionNumbers: ...
def conv_general_dilated(
    lhs: Any,
    rhs: Any,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def conv_general_dilated_local(
    lhs: Any,
    rhs: Any,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    filter_shape: Sequence[int],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    precision: PrecisionLike = None,
) -> Array[IntTuple]: ...
def conv_general_dilated_patches(
    lhs: Any,
    filter_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...
def conv_general_permutations(
    dimension_numbers: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
def conv_general_shape_tuple(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    dimension_numbers: Any,
) -> tuple[int, ...]: ...
def conv_shape_tuple(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[tuple[int, int]],
    batch_group_count: int = 1,
) -> tuple[int, ...]: ...
def conv_transpose(
    lhs: Any,
    rhs: Any,
    strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
    transpose_kernel: bool = False,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    use_consistent_padding: bool = False,
) -> Array[IntTuple]: ...
def conv_transpose_shape_tuple(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    dimension_numbers: Any,
) -> tuple[int, ...]: ...
def conv_with_general_padding(
    lhs: Any,
    rhs: Any,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None,
    rhs_dilation: Sequence[int] | None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...
def custom_linear_solve(
    matvec: Callable[..., Any],
    b: Any,
    solve: Callable[[Callable[..., Any], Any], Any],
    transpose_solve: Callable[[Callable[..., Any], Any], Any] | None = None,
    symmetric: bool = False,
    has_aux: bool = False,
) -> Any: ...
def custom_root(
    f: Callable[..., Any],
    initial_guess: Any,
    solve: Callable[[Callable[..., Any], Any], Any],
    tangent_solve: Callable[[Callable[..., Any], Any], Any],
    has_aux: bool = False,
) -> Any: ...
@overload
def dot[Shape1: _Shape, Shape2: _Shape](
    lhs: Array[Shape1],
    rhs: Array[Shape2],
    *,
    dimension_numbers: None = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
) -> Array[dot_shape(Shape1, Shape2)]: ...
@overload
def dot(
    lhs: Any,
    rhs: Any,
    *,
    dimension_numbers: Any = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def ragged_dot[M: IntVar, K: IntVar, G: IntVar, N: IntVar](
    lhs: Array[[M, K]],
    rhs: Array[[G, K, N]],
    group_sizes: Array[[G]],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: Any = None,
) -> Array[[M, N]]: ...
@overload
def ragged_dot(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def ragged_dot_general(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    ragged_dot_dimension_numbers: Any,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def scaled_dot(
    lhs: Any,
    rhs: Any,
    *,
    lhs_scale: Any = None,
    rhs_scale: Any = None,
    dimension_numbers: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...

# Data types & bitcasting
def bitcast_convert_type(
    operand: Any,
    new_dtype: DTypeLike,
) -> Array[IntTuple]: ...
@overload
def convert_element_type[Shape: _Shape](
    operand: Array[Shape],
    new_dtype: DTypeLike,
) -> Array[Shape]: ...
@overload
def convert_element_type(
    operand: _Scalar | bool,
    new_dtype: DTypeLike,
) -> Array[[]]: ...
@overload
def convert_element_type(
    operand: Any,
    new_dtype: DTypeLike,
) -> Array[IntTuple]: ...

# Reductions, Scans & Window Operations
@overload
def argmax[Shape: _Shape, Axis: Flag[int]](
    operand: Array[Shape],
    axis: Axis,
    index_dtype: DTypeLike = ...,
) -> Array[lax_axis_reduce_shape(Shape, Axis)]: ...
@overload
def argmax(
    operand: Any,
    axis: int,
    index_dtype: DTypeLike = ...,
) -> Array[IntTuple]: ...
@overload
def argmin[Shape: _Shape, Axis: Flag[int]](
    operand: Array[Shape],
    axis: Axis,
    index_dtype: DTypeLike = ...,
) -> Array[lax_axis_reduce_shape(Shape, Axis)]: ...
@overload
def argmin(
    operand: Any,
    axis: int,
    index_dtype: DTypeLike = ...,
) -> Array[IntTuple]: ...
@overload
def associative_scan[Shape: _Shape, Axis: Flag[int] = 0](
    fn: Callable[[Any, Any], Any],
    elems: Array[Shape],
    reverse: bool = False,
    axis: Axis = 0,
) -> Array[lax_associative_scan_shape(Shape, Axis)]: ...
@overload
def associative_scan(
    fn: Callable[[Any, Any], Any],
    elems: Any,
    reverse: bool = False,
    axis: int = 0,
) -> Any: ...
@overload
def cumlogsumexp[Shape: _Shape, Axis: Flag[int] = 0](
    operand: Array[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> Array[lax_scan_shape(Shape, Axis)]: ...
@overload
def cumlogsumexp(
    operand: Any,
    axis: int = 0,
    reverse: bool = False,
) -> Array[IntTuple]: ...
@overload
def cummax[Shape: _Shape, Axis: Flag[int] = 0](
    operand: Array[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> Array[lax_scan_shape(Shape, Axis)]: ...
@overload
def cummax(
    operand: Any,
    axis: int = 0,
    reverse: bool = False,
) -> Array[IntTuple]: ...
@overload
def cummin[Shape: _Shape, Axis: Flag[int] = 0](
    operand: Array[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> Array[lax_scan_shape(Shape, Axis)]: ...
@overload
def cummin(
    operand: Any,
    axis: int = 0,
    reverse: bool = False,
) -> Array[IntTuple]: ...
@overload
def cumprod[Shape: _Shape, Axis: Flag[int] = 0](
    operand: Array[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> Array[lax_scan_shape(Shape, Axis)]: ...
@overload
def cumprod(
    operand: Any,
    axis: int = 0,
    reverse: bool = False,
) -> Array[IntTuple]: ...
@overload
def cumsum[Shape: _Shape, Axis: Flag[int] = 0](
    operand: Array[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> Array[lax_scan_shape(Shape, Axis)]: ...
@overload
def cumsum(
    operand: Any,
    axis: int = 0,
    reverse: bool = False,
) -> Array[IntTuple]: ...
@overload
def reduce[Shape: _Shape, Dims: Flag[tuple[int, ...]]](
    operands: Array[Shape],
    init_values: Any,
    computation: Callable[[Any, Any], Any],
    dimensions: Dims,
    out_sharding: Any = None,
) -> Array[lax_reduce_shape(Shape, Dims)]: ...
@overload
def reduce(
    operands: Any,
    init_values: Any,
    computation: Callable[..., Any],
    dimensions: Sequence[int],
    out_sharding: Any = None,
) -> Any: ...
@overload
def reduce_and[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_and(
    operand: Any,
    axes: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def reduce_max[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
    *,
    out_sharding: Any = None,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_max(
    operand: Any,
    axes: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def reduce_min[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
    *,
    out_sharding: Any = None,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_min(
    operand: Any,
    axes: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def reduce_or[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_or(
    operand: Any,
    axes: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def reduce_precision[Shape: _Shape](
    operand: Array[Shape],
    exponent_bits: int,
    mantissa_bits: int,
) -> Array[Shape]: ...
@overload
def reduce_precision(
    operand: float | int,
    exponent_bits: int,
    mantissa_bits: int,
) -> Array[[]]: ...
@overload
def reduce_precision(
    operand: Any,
    exponent_bits: int,
    mantissa_bits: int,
) -> Array[IntTuple]: ...
@overload
def reduce_prod[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_prod(
    operand: Any,
    axes: Sequence[int],
) -> Array[IntTuple]: ...
@overload
def reduce_sum[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
    *,
    out_sharding: Any = None,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_sum(
    operand: Any,
    axes: Sequence[int],
    *,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def reduce_window[Shape: _Shape](
    operand: Array[Shape],
    init_value: Any,
    computation: Callable[..., Any],
    window_dimensions: Sequence[int],
    window_strides: Sequence[int] | None = None,
    padding: str | Sequence[tuple[int, int]] = "VALID",
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> Array[IntTuple]: ...
@overload
def reduce_window(
    operand: Any,
    init_value: Any,
    computation: Callable[..., Any],
    window_dimensions: Sequence[int],
    window_strides: Sequence[int] | None = None,
    padding: str | Sequence[tuple[int, int]] = "VALID",
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> Any: ...
def reduce_window_shape_tuple(
    operand_shape: Sequence[int],
    window_dimensions: Sequence[int],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> tuple[int, ...]: ...
@overload
def reduce_xor[Shape: _Shape, Axes: Flag[tuple[int, ...]]](
    operand: Array[Shape],
    axes: Axes,
) -> Array[lax_reduce_shape(Shape, Axes)]: ...
@overload
def reduce_xor(
    operand: Any,
    axes: Sequence[int],
) -> Array[IntTuple]: ...

# Selection, Sorting & Searching
@overload
def approx_max_k[Shape: _Shape, K: Flag[int], Dim: Flag[int] = -1](
    operand: Array[Shape],
    k: K,
    reduction_dimension: Dim = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[
    Array[top_k_shape(Shape, K, Dim)],
    Array[top_k_shape(Shape, K, Dim)],
]: ...
@overload
def approx_max_k(
    operand: Any,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def approx_min_k[Shape: _Shape, K: Flag[int], Dim: Flag[int] = -1](
    operand: Array[Shape],
    k: K,
    reduction_dimension: Dim = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[
    Array[top_k_shape(Shape, K, Dim)],
    Array[top_k_shape(Shape, K, Dim)],
]: ...
@overload
def approx_min_k(
    operand: Any,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def clamp[Shape: _Shape](
    min: _Scalar,
    x: Array[Shape],
    max: _Scalar,
) -> Array[Shape]: ...
@overload
def clamp[Shape: _Shape, ShapeMin: _Shape](
    min: Array[ShapeMin],
    x: Array[Shape],
    max: _Scalar,
) -> Array[lax_clamp_max_scalar_shape(ShapeMin, Shape)]: ...
@overload
def clamp[Shape: _Shape, ShapeMax: _Shape](
    min: _Scalar,
    x: Array[Shape],
    max: Array[ShapeMax],
) -> Array[lax_clamp_min_scalar_shape(Shape, ShapeMax)]: ...
@overload
def clamp[Shape: _Shape, ShapeMin: _Shape, ShapeMax: _Shape](
    min: Array[ShapeMin],
    x: Array[Shape],
    max: Array[ShapeMax],
) -> Array[lax_clamp_shape(ShapeMin, Shape, ShapeMax)]: ...
@overload
def clamp(
    min: _Scalar,
    x: _Scalar,
    max: _Scalar,
) -> Array[[]]: ...
@overload
def clamp(
    min: Any,
    x: Any,
    max: Any,
) -> Array[IntTuple]: ...
@overload
def select[Shape: _Shape, Shape2: _Shape](
    pred: bool | int,
    on_true: Array[Shape],
    on_false: Array[Shape2],
) -> Array[lax_select_scalar_pred_shape(Shape, Shape2)]: ...
@overload
def select[Shape: _Shape, Shape2: _Shape, PredShape: _Shape](
    pred: Array[PredShape],
    on_true: Array[Shape],
    on_false: Array[Shape2],
) -> Array[lax_select_shape(PredShape, Shape, Shape2)]: ...
@overload
def select(
    pred: bool | int,
    on_true: _Scalar,
    on_false: _Scalar,
) -> Array[[]]: ...
@overload
def select(
    pred: Any,
    on_true: Any,
    on_false: Any,
) -> Array[IntTuple]: ...
@overload
def select_n[Shape: _Shape, WhichShape: _Shape](
    which: Array[WhichShape],
    *cases: Array[Shape],
) -> Array[lax_select_n_shape(WhichShape, Shape)]: ...
@overload
def select_n[Shape: _Shape](
    which: bool | int,
    *cases: Array[Shape],
) -> Array[Shape]: ...
@overload
def select_n(
    which: bool | int,
    *cases: _Scalar,
) -> Array[[]]: ...
@overload
def select_n(
    which: Any,
    *cases: Any,
) -> Array[IntTuple]: ...
@overload
def sort[Shape: _Shape, Dim: Flag[int] = -1](
    operand: Array[Shape],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> Array[lax_sort_shape(Shape, Dim)]: ...
@overload
def sort[Shape: _Shape, Dim: Flag[int] = -1](
    operand: tuple[Array[Shape], Array[Shape]],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> tuple[Array[lax_sort_shape(Shape, Dim)], Array[lax_sort_shape(Shape, Dim)]]: ...
@overload
def sort[Shape: _Shape, Dim: Flag[int] = -1](
    operand: Sequence[Array[Shape]],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> tuple[Array[lax_sort_shape(Shape, Dim)], ...]: ...
@overload
def sort(
    operand: Any,
    dimension: int = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> Any: ...
@overload
def sort_key_val[Shape1: _Shape, Shape2: _Shape, Dim: Flag[int] = -1](
    keys: Array[Shape1],
    values: Array[Shape2],
    dimension: Dim = -1,
    is_stable: bool = True,
) -> tuple[
    Array[lax_sort_key_val_shape(Shape1, Shape2, Dim)],
    Array[lax_sort_key_val_shape(Shape1, Shape2, Dim)],
]: ...
@overload
def sort_key_val(
    keys: Any,
    values: Any,
    dimension: int = -1,
    is_stable: bool = True,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def top_k[Shape: _Shape, K: Flag[int], Axis: Flag[int] = -1](
    operand: Array[Shape],
    k: K,
    *,
    axis: Axis = -1,
    is_stable: bool = True,
) -> tuple[
    Array[top_k_shape(Shape, K, Axis)],
    Array[top_k_shape(Shape, K, Axis)],
]: ...
@overload
def top_k(
    operand: Any,
    k: int,
    *,
    axis: int = -1,
    is_stable: bool = True,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...

# -----------------------------------------------------------------------------
# Control Flow & Higher-Order Functions
# -----------------------------------------------------------------------------

@overload
def cond[InVal, OutVal](
    pred: bool | Array[[]],
    true_fun: Callable[[InVal], OutVal],
    false_fun: Callable[[InVal], OutVal],
    *,
    operand: InVal,
) -> OutVal: ...
@overload
def cond[*InVals, OutVal](
    pred: bool | Array[[]],
    true_fun: Callable[[*InVals], OutVal],
    false_fun: Callable[[*InVals], OutVal],
    *operands: *InVals,
) -> OutVal: ...
def fori_loop[T](
    lower: int | Array[[]],
    upper: int | Array[[]],
    body_fun: Callable[[int | Array[[]], T], T],
    init_val: T,
    *,
    unroll: int | bool | None = None,
) -> T: ...
@overload
def map[N: IntVar, InShape: _Shape, OutShape: _Shape](
    f: Callable[[Array[InShape]], Array[OutShape]],
    xs: Array[[N, *Elements[InShape]]],
    *,
    batch_size: int | None = None,
) -> Array[[N, *Elements[OutShape]]]: ...
@overload
def map(
    f: Callable[..., Any],
    xs: Any,
    *,
    batch_size: int | None = None,
) -> Any: ...
@overload
def scan[Carry, N: IntVar, InShape: _Shape, OutShape: _Shape](
    f: Callable[[Carry, Array[InShape]], tuple[Carry, Array[OutShape]]],
    init: Carry,
    xs: Array[[N, *Elements[InShape]]],
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Array[[N, *Elements[OutShape]]]]: ...
@overload
def scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]: ...
@overload
def scan[Carry, Y](
    f: Callable[[Carry, Any], tuple[Carry, Y]],
    init: Carry,
    xs: None = None,
    *,
    length: int,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]: ...
@overload
def scan(
    f: Callable[..., Any],
    init: Any,
    xs: Any = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Any, Any]: ...
@overload
def switch[InVal, OutVal](
    index: int | Array[[]],
    branches: Sequence[Callable[[InVal], OutVal]],
    *,
    operand: InVal,
) -> OutVal: ...
@overload
def switch[*InVals, OutVal](
    index: int | Array[[]],
    branches: Sequence[Callable[[*InVals], OutVal]],
    *operands: *InVals,
) -> OutVal: ...
def while_loop[T](
    cond_fun: Callable[[T], bool | Array[[]]],
    body_fun: Callable[[T], T],
    init_val: T,
) -> T: ...

# -----------------------------------------------------------------------------
# Parallel & Collective Operations
# -----------------------------------------------------------------------------

@overload
def all_gather[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
    axis: int = 0,
    tiled: bool = False,
    to: str = "varying",
) -> Array[IntTuple]: ...
@overload
def all_gather[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
    axis: int = 0,
    tiled: bool = False,
    to: str = "varying",
) -> T: ...
@overload
def all_to_all[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    split_axis: int,
    concat_axis: int,
    *,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> Array[IntTuple]: ...
@overload
def all_to_all[T](
    x: T,
    axis_name: Hashable,
    split_axis: int,
    concat_axis: int,
    *,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> T: ...
def axis_index(axis_name: Hashable) -> Array[[]]: ...
def axis_size(axis_name: Hashable) -> int: ...
@overload
def pbroadcast[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    source: int,
) -> Array[Shape]: ...
@overload
def pbroadcast[T](
    x: T,
    axis_name: Hashable,
    source: int,
) -> T: ...
@overload
def pcast[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    to: str,
) -> Array[Shape]: ...
@overload
def pcast[T](
    x: T,
    axis_name: Hashable,
    *,
    to: str,
) -> T: ...
@overload
def pmax[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> Array[Shape]: ...
@overload
def pmax[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def pmean[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> Array[Shape]: ...
@overload
def pmean[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def pmin[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> Array[Shape]: ...
@overload
def pmin[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def ppermute[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    perm: Sequence[tuple[int, int]],
) -> Array[Shape]: ...
@overload
def ppermute[T](
    x: T,
    axis_name: Hashable,
    perm: Sequence[tuple[int, int]],
) -> T: ...
def precv(
    token: Any,
    out_shape: Any,
    axis_name: Hashable,
    perm: Sequence[tuple[int, int]],
) -> Any: ...
def psend(
    x: Any,
    axis_name: Hashable,
    perm: Sequence[tuple[int, int]],
) -> Any: ...
@overload
def pshuffle[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    perm: Sequence[int],
) -> Array[Shape]: ...
@overload
def pshuffle[T](
    x: T,
    axis_name: Hashable,
    perm: Sequence[int],
) -> T: ...
@overload
def psum[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> Array[Shape]: ...
@overload
def psum[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def psum_scatter[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    *,
    scatter_dimension: int = 0,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> Array[IntTuple]: ...
@overload
def psum_scatter[T](
    x: T,
    axis_name: Hashable,
    *,
    scatter_dimension: int = 0,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> T: ...
@overload
def pswapaxes[Shape: _Shape](
    x: Array[Shape],
    axis_name: Hashable,
    axis: int,
    *,
    axis_index_groups: Any = None,
) -> Array[Shape]: ...
@overload
def pswapaxes[T](
    x: T,
    axis_name: Hashable,
    axis: int,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def ragged_all_to_all[OutputShape: _Shape](
    operand: Any,
    output: Array[OutputShape],
    input_offsets: Any,
    send_sizes: Any,
    output_offsets: Any,
    recv_sizes: Any,
    *,
    axis_name: Hashable,
    axis_index_groups: Any = None,
) -> Array[OutputShape]: ...
@overload
def ragged_all_to_all(
    operand: Any,
    output: Any,
    input_offsets: Any,
    send_sizes: Any,
    output_offsets: Any,
    recv_sizes: Any,
    *,
    axis_name: Hashable,
    axis_index_groups: Any = None,
) -> Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Data Types, Bitcasting & RNG
# -----------------------------------------------------------------------------

def dtype(x: Any) -> np.dtype: ...
@overload
def rng_bit_generator[KeyShape: _Shape, Shape: _Shape](
    key: Array[KeyShape],
    shape: Shape,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: Any = None,
) -> tuple[Array[KeyShape], Array[Shape]]: ...
@overload
def rng_bit_generator[KeyShape: _Shape](
    key: Array[KeyShape],
    shape: Sequence[int] | int,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: Any = None,
) -> tuple[Array[KeyShape], Array[IntTuple]]: ...
@overload
def rng_bit_generator(
    key: Any,
    shape: Any,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: Any = None,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def rng_uniform[Shape: _Shape](
    a: float | Array[[]],
    b: float | Array[[]],
    shape: Shape,
) -> Array[Shape]: ...
@overload
def rng_uniform(
    a: float | Array[[]],
    b: float | Array[[]],
    shape: Sequence[int] | int,
) -> Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Compiler, Token & Miscellaneous
# -----------------------------------------------------------------------------

def after_all(*operands: Any) -> Any: ...
def composite[F: Callable[..., Any]](
    decomposition: F,
    name: str,
    version: int = 0,
) -> F: ...
def create_token(_: Any = None) -> Any: ...
def dce_sink(val: Any, *, prevent_mlir_dce: bool = False) -> None: ...
def optimization_barrier[T](operand: T, /) -> T: ...
def platform_dependent[*Args, T](
    *args: *Args,
    default: Callable[[*Args], T] | None = None,
    **per_platform: Callable[[*Args], T],
) -> T: ...
def shape_as_value(shape: Any) -> Array[IntTuple]: ...
def stage[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def stop_gradient[T](x: T) -> T: ...
def with_sharding_constraint[T](x: T, shardings: Any) -> T: ...
