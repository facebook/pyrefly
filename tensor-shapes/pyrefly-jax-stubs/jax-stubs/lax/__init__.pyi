# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Hashable, overload, Sequence

import numpy as np
from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._shapes import (
    broadcast_to_rank_shape,
    collapse_shape,
    collapse_to_end_shape,
    concatenate_shape,
    dot_shape,
    lax_associative_scan_shape,
    lax_axis_reduce_shape,
    lax_broadcast,
    lax_clamp_shape,
    lax_dynamic_index_in_dim_shape,
    lax_dynamic_slice_in_dim_shape,
    lax_dynamic_slice_shape,
    lax_reduce_shape,
    lax_scan_shape,
    lax_select_n_shape,
    lax_select_shape,
    lax_sort_key_val_shape,
    lax_sort_shape,
    lax_squeeze_shape,
    permute_shape,
    shape_as_value_shape,
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
from jax._src.sharding_impls import (
    NamedSharding as _NamedSharding,
    PartitionSpec as _PartitionSpec,
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
type _ShapedArrayLike[Shape: _Shape] = _Array[Shape] | np.ndarray[Shape]

# Unary elementwise operators
def abs[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def acos[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def acosh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def asin[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def asinh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def atan[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def atanh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def bessel_i0e[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def bessel_i1e[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def bitwise_not[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def cbrt[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def ceil[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def clz[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def conj[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def cos[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def cosh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def digamma[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def erf[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def erf_inv[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def erfc[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def exp[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def exp2[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def expm1[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def floor[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def imag[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def integer_pow[Shape: _Shape = []](x: _ArrayLike[Shape], y: int) -> _Array[Shape]: ...
def is_finite[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def lgamma[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def log[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def log1p[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def logistic[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def neg[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def population_count[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def real[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def reciprocal[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def round[Shape: _Shape = []](
    x: _ArrayLike[Shape], rounding_method: Any = ...
) -> _Array[Shape]: ...
def rsqrt[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def sign[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def sin[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def sinh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def sqrt[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def square[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def tan[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...
def tanh[Shape: _Shape = []](
    x: _ArrayLike[Shape], *, accuracy: Any = None
) -> _Array[Shape]: ...

# Binary elementwise operators with strict rank-matching broadcasting
def add[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def atan2[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def bitwise_and[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def bitwise_or[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def bitwise_xor[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def complex[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def div[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def eq[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def ge[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def gt[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def igamma[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1], x: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def igamma_grad_a[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1], x: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def igammac[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1], x: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def le[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def lt[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def max[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def min[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def mul[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1],
    y: _ArrayLike[Shape2],
    /,
    *,
    out_dtype: DTypeLike | None = None,
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def mulhi[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def ne[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def nextafter[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def polygamma[Shape1: _Shape = [], Shape2: _Shape = []](
    m: _ArrayLike[Shape1], x: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def pow[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def rem[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def shift_left[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def shift_right_arithmetic[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def shift_right_logical[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def sub[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], y: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
def zeta[Shape1: _Shape = [], Shape2: _Shape = []](
    x: _ArrayLike[Shape1], q: _ArrayLike[Shape2], /
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...
@overload
def betainc[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    b: _ArrayLike[Shape],
    x: _ArrayLike[Shape],
    /,
) -> _Array[Shape]: ...
@overload
def betainc[
    Shape1: _Shape = [],
    Shape2: _Shape = [],
    Shape3: _Shape = [],
](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    x: _ArrayLike[Shape3],
    /,
) -> _Array[lax_broadcast(lax_broadcast(Shape1, Shape2), Shape3)]: ...
@overload
def fft[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    fft_type: FftType | str,
    fft_lengths: Sequence[int],
) -> _Array[IntTuple]: ...
@overload
def fft(
    x: Any,
    fft_type: FftType | str,
    fft_lengths: Sequence[int],
) -> _Array[IntTuple]: ...
@overload
def random_gamma_grad[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    x: _ArrayLike[Shape],
    /,
) -> _Array[Shape]: ...
@overload
def random_gamma_grad[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1],
    x: _ArrayLike[Shape2],
    /,
) -> _Array[lax_broadcast(Shape1, Shape2)]: ...

# -----------------------------------------------------------------------------
# Array Creation & Constants
# -----------------------------------------------------------------------------

@overload
def broadcasted_iota[Shape: _Shape](
    dtype: DTypeLike,
    shape: Shape,
    dimension: int,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def broadcasted_iota(
    dtype: DTypeLike,
    shape: Sequence[int] | int,
    dimension: int,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def empty(
    shape: tuple[()],
    dtype: DTypeLike,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[]]: ...
@overload
def empty[N: IntVar](
    shape: Int[N],
    dtype: DTypeLike,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape,
    dtype: DTypeLike,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int,
    dtype: DTypeLike,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def full(
    shape: tuple[()],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> _Array[[]]: ...
@overload
def full[N: IntVar](
    shape: Int[N],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> _Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> _Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    sharding: Any = None,
) -> _Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: None = None,
    *,
    sharding: Any = None,
) -> _Array[Shape]: ...
@overload
def full_like[N: IntVar](
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Int[N] = ...,
    *,
    sharding: Any = None,
) -> _Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Shape = ...,
    *,
    sharding: Any = None,
) -> _Array[Shape]: ...
@overload
def full_like(
    x: Any,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    shape: Sequence[int] | int | None = None,
    *,
    sharding: Any = None,
) -> _Array[IntTuple]: ...
@overload
def iota[N: IntVar](dtype: DTypeLike, size: Int[N]) -> _Array[[N]]: ...
@overload
def iota(dtype: DTypeLike, size: int) -> _Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Shape Manipulation, Slicing & Reshaping
# -----------------------------------------------------------------------------

@overload
def broadcast[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    sizes: tuple[()],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def broadcast[D0: IntVar, Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    sizes: tuple[Int[D0]],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[D0, *Elements[Shape]]]: ...
@overload
def broadcast[D0: IntVar, D1: IntVar, Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    sizes: tuple[Int[D0], Int[D1]],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[D0, D1, *Elements[Shape]]]: ...
@overload
def broadcast[D0: IntVar, D1: IntVar, D2: IntVar, Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    sizes: tuple[Int[D0], Int[D1], Int[D2]],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[D0, D1, D2, *Elements[Shape]]]: ...
@overload
def broadcast(
    operand: Any,
    sizes: Sequence[int],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def broadcast_in_dim[Shape: _Shape](
    operand: Any,
    shape: Shape,
    broadcast_dimensions: Sequence[int],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def broadcast_in_dim(
    operand: Any,
    shape: Sequence[int] | int,
    broadcast_dimensions: Sequence[int],
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
def broadcast_like[InShape: _Shape = [], OutShape: _Shape = []](
    arr: _ArrayLike[InShape],
    like_arr: _ArrayLike[OutShape],
) -> _Array[OutShape]: ...
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
def broadcast_to_rank[Rank: Flag[int], Shape: _Shape = []](
    x: _ArrayLike[Shape],
    rank: Rank,
) -> _Array[broadcast_to_rank_shape(Shape, Rank)]: ...
@overload
def collapse[Start: Flag[int], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    start_dimension: Start,
    stop_dimension: None = None,
) -> _Array[collapse_to_end_shape(Shape, Start)]: ...
@overload
def collapse[Start: Flag[int], Stop: Flag[int], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    start_dimension: Start,
    stop_dimension: Stop,
) -> _Array[collapse_shape(Shape, Start, Stop)]: ...
@overload
def collapse(
    operand: Any,
    start_dimension: int,
    stop_dimension: int | None = None,
) -> _Array[IntTuple]: ...
@overload
def concatenate[Shapes: IntTuples, Dimension: Flag[int] = 0](
    operands: MapIntTuples[lambda S: _Array[S], Shapes],
    dimension: Dimension = 0,
) -> _Array[concatenate_shape(Shapes, Dimension)]: ...
@overload
def concatenate(
    operands: Any,
    dimension: int = 0,
) -> _Array[IntTuple]: ...
def expand_dims(
    array: Any,
    dimensions: Sequence[int],
) -> _Array[IntTuple]: ...
def pad(
    operand: Any,
    padding_value: Any,
    padding_config: Sequence[tuple[int, int, int]],
) -> _Array[IntTuple]: ...
def padtype_to_pads(
    in_shape: Sequence[int],
    window_shape: Sequence[int],
    window_strides: Sequence[int],
    padding: str,
) -> list[tuple[int, int]]: ...
def reshape[NewShape: _Shape, Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    new_sizes: NewShape,
    dimensions: Sequence[int] | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[NewShape]: ...
def rev[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    dimensions: Sequence[int],
) -> _Array[Shape]: ...
def slice[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None = None,
) -> _Array[IntTuple]: ...
def slice_in_dim(
    operand: Any,
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    axis: int = 0,
) -> _Array[IntTuple]: ...
def split(
    operand: Any,
    sizes: Sequence[int],
    axis: int = 0,
) -> list[_Array[IntTuple]]: ...
def squeeze[Dims: Flag[tuple[int, ...]], Shape: _Shape = []](
    array: _ArrayLike[Shape],
    dimensions: Dims,
) -> _Array[lax_squeeze_shape(Shape, Dims)]: ...
@overload
def stack[Shapes: IntTuples, Axis: Flag[int] = 0](
    operands: MapIntTuples[lambda S: _Array[S], Shapes],
    axis: Axis = 0,
) -> _Array[stack_shape(Shapes, Axis)]: ...
@overload
def stack(
    operands: Any,
    axis: int = 0,
) -> _Array[IntTuple]: ...
def tile[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    reps: Sequence[int],
) -> _Array[IntTuple]: ...
def transpose[Permutation: Flag[_Axis], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    permutation: Permutation,
) -> _Array[permute_shape(Shape, Permutation)]: ...
def unstack[Batch: IntTuple, M: IntVar](
    x: _ShapedArrayLike[[*Elements[Batch], M]],
    axis: int = 0,
) -> tuple[_Array[IntTuple], ...]: ...

# Dynamic Slicing & Gather/Scatter

@overload
def dynamic_index_in_dim[
    Shape: _Shape = [],
    Axis: Flag[int] = 0,
    KeepDims: Flag[bool] = True,
](
    operand: _ArrayLike[Shape],
    index: Any,
    axis: Axis = 0,
    keepdims: KeepDims = True,
    *,
    allow_negative_indices: bool = True,
) -> _Array[lax_dynamic_index_in_dim_shape(Shape, Axis, KeepDims)]: ...
@overload
def dynamic_index_in_dim(
    operand: Any,
    index: Any,
    axis: int = 0,
    keepdims: bool = True,
    *,
    allow_negative_indices: bool = True,
) -> _Array[IntTuple]: ...
@overload
def dynamic_slice[SliceSizes: _Shape, Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    start_indices: Any,
    slice_sizes: SliceSizes,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> _Array[lax_dynamic_slice_shape(Shape, SliceSizes)]: ...
@overload
def dynamic_slice(
    operand: Any,
    start_indices: Any,
    slice_sizes: Sequence[int],
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> _Array[IntTuple]: ...
@overload
def dynamic_slice_in_dim[
    SliceSize: Flag[int],
    Shape: _Shape = [],
    Axis: Flag[int] = 0,
](
    operand: _ArrayLike[Shape],
    start_index: Any,
    slice_size: SliceSize,
    axis: Axis = 0,
    *,
    allow_negative_indices: bool = True,
) -> _Array[lax_dynamic_slice_in_dim_shape(Shape, SliceSize, Axis)]: ...
@overload
def dynamic_slice_in_dim(
    operand: Any,
    start_index: Any,
    slice_size: int,
    axis: int = 0,
    *,
    allow_negative_indices: bool = True,
) -> _Array[IntTuple]: ...
@overload
def dynamic_update_index_in_dim[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    update: Any,
    index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> _Array[Shape]: ...
@overload
def dynamic_update_index_in_dim(
    operand: Any,
    update: Any,
    index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> _Array[IntTuple]: ...
@overload
def dynamic_update_slice[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    update: Any,
    start_indices: Any,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> _Array[Shape]: ...
@overload
def dynamic_update_slice(
    operand: Any,
    update: Any,
    start_indices: Any,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> _Array[IntTuple]: ...
@overload
def dynamic_update_slice_in_dim[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    update: Any,
    start_index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> _Array[Shape]: ...
@overload
def dynamic_update_slice_in_dim(
    operand: Any,
    update: Any,
    start_index: Any,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> _Array[IntTuple]: ...
def gather(
    operand: _ArrayLike[Any],
    start_indices: _ArrayLike[Any],
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Sequence[int],
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: str | GatherScatterMode | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
@overload
def index_in_dim[Shape: _Shape = [], Axis: Flag[int] = 0, KeepDims: Flag[bool] = True](
    operand: _ArrayLike[Shape],
    index: int,
    axis: Axis = 0,
    keepdims: KeepDims = True,
) -> _Array[lax_dynamic_index_in_dim_shape(Shape, Axis, KeepDims)]: ...
@overload
def index_in_dim(
    operand: Any,
    index: int,
    axis: int = 0,
    keepdims: bool = True,
) -> _Array[IntTuple]: ...
def index_take(
    src: _ArrayLike[Any],
    idxs: _ArrayLike[Any],
    axes: Sequence[int],
) -> _Array[IntTuple]: ...
@overload
def scatter[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_add[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_apply[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    func: Callable[[Any], Any],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    update_shape: Sequence[int] = (),
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_max[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_min[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_mul[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...
@overload
def scatter_sub[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    scatter_indices: Any,
    updates: Any,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> _Array[Shape]: ...
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
) -> _Array[IntTuple]: ...

# Linear Algebra, Contractions & Convolutions

@overload
def batch_matmul[Batch: IntTuple, M: IntVar, K: IntVar, N: IntVar](
    lhs: _ShapedArrayLike[[*Elements[Batch], M, K]],
    rhs: _ShapedArrayLike[[*Elements[Batch], K, N]],
    precision: PrecisionLike = None,
) -> _Array[[*Elements[Batch], M, N]]: ...
@overload
def batch_matmul(
    lhs: Any,
    rhs: Any,
    precision: PrecisionLike = None,
) -> _Array[IntTuple]: ...
def conv(
    lhs: Any,
    rhs: Any,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
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
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
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
) -> _Array[IntTuple]: ...
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
) -> _Array[IntTuple]: ...
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
) -> _Array[IntTuple]: ...
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
) -> _Array[IntTuple]: ...
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
def dot[Shape1: _Shape = [], Shape2: _Shape = []](
    lhs: _ArrayLike[Shape1],
    rhs: _ArrayLike[Shape2],
    *,
    dimension_numbers: None = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[dot_shape(Shape1, Shape2)]: ...
@overload
def dot(
    lhs: Any,
    rhs: Any,
    *,
    dimension_numbers: Any = None,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
def dot_general(
    lhs: Any,
    rhs: Any,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def ragged_dot[M: IntVar, K: IntVar, G: IntVar, N: IntVar](
    lhs: _ShapedArrayLike[[M, K]],
    rhs: _ShapedArrayLike[[G, K, N]],
    group_sizes: _ShapedArrayLike[[G]],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[M, N]]: ...
@overload
def ragged_dot(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
def ragged_dot_general(
    lhs: Any,
    rhs: Any,
    group_sizes: Any,
    ragged_dot_dimension_numbers: Any,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
def scaled_dot(
    lhs: Any,
    rhs: Any,
    *,
    lhs_scale: Any = None,
    rhs_scale: Any = None,
    dimension_numbers: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[IntTuple]: ...

# Data types & bitcasting
def bitcast_convert_type[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    new_dtype: DTypeLike,
) -> _Array[IntTuple]: ...
def convert_element_type[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    new_dtype: DTypeLike,
) -> _Array[Shape]: ...

# Reductions, Scans & Window Operations
def argmax[Axis: Flag[int], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axis: Axis,
    index_dtype: DTypeLike = ...,
) -> _Array[lax_axis_reduce_shape(Shape, Axis)]: ...
def argmin[Axis: Flag[int], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axis: Axis,
    index_dtype: DTypeLike = ...,
) -> _Array[lax_axis_reduce_shape(Shape, Axis)]: ...
def associative_scan[Shape: _Shape = [], Axis: Flag[int] = 0](
    fn: Callable[[Any, Any], Any],
    elems: _ArrayLike[Shape],
    reverse: bool = False,
    axis: Axis = 0,
) -> _Array[lax_associative_scan_shape(Shape, Axis)]: ...
def cumlogsumexp[Shape: _Shape = [], Axis: Flag[int] = 0](
    operand: _ArrayLike[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> _Array[lax_scan_shape(Shape, Axis)]: ...
def cummax[Shape: _Shape = [], Axis: Flag[int] = 0](
    operand: _ArrayLike[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> _Array[lax_scan_shape(Shape, Axis)]: ...
def cummin[Shape: _Shape = [], Axis: Flag[int] = 0](
    operand: _ArrayLike[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> _Array[lax_scan_shape(Shape, Axis)]: ...
def cumprod[Shape: _Shape = [], Axis: Flag[int] = 0](
    operand: _ArrayLike[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> _Array[lax_scan_shape(Shape, Axis)]: ...
def cumsum[Shape: _Shape = [], Axis: Flag[int] = 0](
    operand: _ArrayLike[Shape],
    axis: Axis = 0,
    reverse: bool = False,
) -> _Array[lax_scan_shape(Shape, Axis)]: ...
def reduce[Dims: Flag[tuple[int, ...]], Shape: _Shape = []](
    operands: _ArrayLike[Shape],
    init_values: Any,
    computation: Callable[[Any, Any], Any],
    dimensions: Dims,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[lax_reduce_shape(Shape, Dims)]: ...
def reduce_and[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_max[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_min[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_or[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_precision[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    exponent_bits: int,
    mantissa_bits: int,
) -> _Array[Shape]: ...
def reduce_prod[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_sum[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...
def reduce_window[Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    init_value: Any,
    computation: Callable[..., Any],
    window_dimensions: Sequence[int],
    window_strides: Sequence[int] | None = None,
    padding: str | Sequence[tuple[int, int]] = "VALID",
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> _Array[IntTuple]: ...
def reduce_window_shape_tuple(
    operand_shape: Sequence[int],
    window_dimensions: Sequence[int],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> tuple[int, ...]: ...
def reduce_xor[Axes: Flag[tuple[int, ...]], Shape: _Shape = []](
    operand: _ArrayLike[Shape],
    axes: Axes,
) -> _Array[lax_reduce_shape(Shape, Axes)]: ...

# Selection, Sorting & Searching
def approx_max_k[K: Flag[int], Shape: _Shape = [], Dim: Flag[int] = -1](
    operand: _ArrayLike[Shape],
    k: K,
    reduction_dimension: Dim = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[
    _Array[top_k_shape(Shape, K, Dim)],
    _Array[top_k_shape(Shape, K, Dim)],
]: ...
def approx_min_k[K: Flag[int], Shape: _Shape = [], Dim: Flag[int] = -1](
    operand: _ArrayLike[Shape],
    k: K,
    reduction_dimension: Dim = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
) -> tuple[
    _Array[top_k_shape(Shape, K, Dim)],
    _Array[top_k_shape(Shape, K, Dim)],
]: ...
def clamp[
    ShapeMin: _Shape = [],
    Shape: _Shape = [],
    ShapeMax: _Shape = [],
](
    min: _ArrayLike[ShapeMin],
    x: _ArrayLike[Shape],
    max: _ArrayLike[ShapeMax],
) -> _Array[lax_clamp_shape(ShapeMin, Shape, ShapeMax)]: ...
def select[
    PredShape: _Shape = [],
    Shape: _Shape = [],
    Shape2: _Shape = [],
](
    pred: _ArrayLike[PredShape],
    on_true: _ArrayLike[Shape],
    on_false: _ArrayLike[Shape2],
) -> _Array[lax_select_shape(PredShape, Shape, Shape2)]: ...
def select_n[WhichShape: _Shape = [], Shape: _Shape = []](
    which: _ArrayLike[WhichShape],
    *cases: _ArrayLike[Shape],
) -> _Array[lax_select_n_shape(WhichShape, Shape)]: ...
@overload
def sort[Shape: _Shape = [], Dim: Flag[int] = -1](
    operand: _ArrayLike[Shape],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> _Array[lax_sort_shape(Shape, Dim)]: ...
@overload
def sort[Shape: _Shape = [], Dim: Flag[int] = -1](
    operand: tuple[_ArrayLike[Shape], _ArrayLike[Shape]],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> tuple[_Array[lax_sort_shape(Shape, Dim)], _Array[lax_sort_shape(Shape, Dim)]]: ...
@overload
def sort[Shape: _Shape = [], Dim: Flag[int] = -1](
    operand: Sequence[_ArrayLike[Shape]],
    dimension: Dim = -1,
    is_stable: bool = True,
    num_keys: int = 1,
) -> tuple[_Array[lax_sort_shape(Shape, Dim)], ...]: ...
def sort_key_val[Shape1: _Shape = [], Shape2: _Shape = [], Dim: Flag[int] = -1](
    keys: _ArrayLike[Shape1],
    values: _ArrayLike[Shape2],
    dimension: Dim = -1,
    is_stable: bool = True,
) -> tuple[
    _Array[lax_sort_key_val_shape(Shape1, Shape2, Dim)],
    _Array[lax_sort_key_val_shape(Shape1, Shape2, Dim)],
]: ...
def top_k[K: Flag[int], Shape: _Shape = [], Axis: Flag[int] = -1](
    operand: _ArrayLike[Shape],
    k: K,
    *,
    axis: Axis = -1,
    is_stable: bool = True,
) -> tuple[
    _Array[top_k_shape(Shape, K, Axis)],
    _Array[top_k_shape(Shape, K, Axis)],
]: ...

# -----------------------------------------------------------------------------
# Control Flow & Higher-Order Functions
# -----------------------------------------------------------------------------

@overload
def cond[InVal, OutVal](
    pred: _ArrayLike[[]],
    true_fun: Callable[[InVal], OutVal],
    false_fun: Callable[[InVal], OutVal],
    *,
    operand: InVal,
) -> OutVal: ...
@overload
def cond[*InVals, OutVal](
    pred: _ArrayLike[[]],
    true_fun: Callable[[*InVals], OutVal],
    false_fun: Callable[[*InVals], OutVal],
    *operands: *InVals,
) -> OutVal: ...
def fori_loop[T](
    lower: _ArrayLike[[]],
    upper: _ArrayLike[[]],
    body_fun: Callable[[int | _Array[[]], T], T],
    init_val: T,
    *,
    unroll: int | bool | None = None,
) -> T: ...
@overload
def map[N: IntVar, InShape: _Shape, OutShape: _Shape](
    f: Callable[[_Array[InShape]], _Array[OutShape]],
    xs: _ShapedArrayLike[[N, *Elements[InShape]]],
    *,
    batch_size: int | None = None,
) -> _Array[[N, *Elements[OutShape]]]: ...
@overload
def map(
    f: Callable[..., Any],
    xs: Any,
    *,
    batch_size: int | None = None,
) -> Any: ...
@overload
def scan[Carry, N: IntVar, InShape: _Shape, OutShape: _Shape](
    f: Callable[[Carry, _Array[InShape]], tuple[Carry, _Array[OutShape]]],
    init: Carry,
    xs: _ShapedArrayLike[[N, *Elements[InShape]]],
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, _Array[[N, *Elements[OutShape]]]]: ...
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
    index: _ArrayLike[[]],
    branches: Sequence[Callable[[InVal], OutVal]],
    *,
    operand: InVal,
) -> OutVal: ...
@overload
def switch[*InVals, OutVal](
    index: _ArrayLike[[]],
    branches: Sequence[Callable[[*InVals], OutVal]],
    *operands: *InVals,
) -> OutVal: ...
def while_loop[T](
    cond_fun: Callable[[T], _ArrayLike[[]]],
    body_fun: Callable[[T], T],
    init_val: T,
) -> T: ...

# -----------------------------------------------------------------------------
# Parallel & Collective Operations
# -----------------------------------------------------------------------------

@overload
def all_gather[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
    axis: int = 0,
    tiled: bool = False,
    to: str = "varying",
) -> _Array[IntTuple]: ...
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
def all_to_all[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    split_axis: int,
    concat_axis: int,
    *,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> _Array[IntTuple]: ...
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
def axis_index(axis_name: Hashable) -> _Array[[]]: ...
def axis_size(axis_name: Hashable) -> int: ...
@overload
def pbroadcast[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    source: int,
) -> _Array[Shape]: ...
@overload
def pbroadcast[T](
    x: T,
    axis_name: Hashable,
    source: int,
) -> T: ...
@overload
def pcast[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    to: str,
) -> _Array[Shape]: ...
@overload
def pcast[T](
    x: T,
    axis_name: Hashable,
    *,
    to: str,
) -> T: ...
@overload
def pmax[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> _Array[Shape]: ...
@overload
def pmax[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def pmean[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> _Array[Shape]: ...
@overload
def pmean[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def pmin[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> _Array[Shape]: ...
@overload
def pmin[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def ppermute[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    perm: Sequence[tuple[int, int]],
) -> _Array[Shape]: ...
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
def pshuffle[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    perm: Sequence[int],
) -> _Array[Shape]: ...
@overload
def pshuffle[T](
    x: T,
    axis_name: Hashable,
    perm: Sequence[int],
) -> T: ...
@overload
def psum[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> _Array[Shape]: ...
@overload
def psum[T](
    x: T,
    axis_name: Hashable,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def psum_scatter[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    *,
    scatter_dimension: int = 0,
    axis_index_groups: Any = None,
    tiled: bool = False,
) -> _Array[IntTuple]: ...
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
def pswapaxes[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis_name: Hashable,
    axis: int,
    *,
    axis_index_groups: Any = None,
) -> _Array[Shape]: ...
@overload
def pswapaxes[T](
    x: T,
    axis_name: Hashable,
    axis: int,
    *,
    axis_index_groups: Any = None,
) -> T: ...
@overload
def ragged_all_to_all[OutputShape: _Shape = []](
    operand: Any,
    output: _ArrayLike[OutputShape],
    input_offsets: Any,
    send_sizes: Any,
    output_offsets: Any,
    recv_sizes: Any,
    *,
    axis_name: Hashable,
    axis_index_groups: Any = None,
) -> _Array[OutputShape]: ...
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
) -> _Array[IntTuple]: ...

# -----------------------------------------------------------------------------
# Data Types, Bitcasting & RNG
# -----------------------------------------------------------------------------

def dtype(x: Any) -> np.dtype: ...
@overload
def rng_bit_generator[Shape: _Shape, KeyShape: _Shape = []](
    key: _ArrayLike[KeyShape],
    shape: Shape,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> tuple[_Array[KeyShape], _Array[Shape]]: ...
@overload
def rng_bit_generator[KeyShape: _Shape = []](
    key: _ArrayLike[KeyShape],
    shape: Sequence[int] | int,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> tuple[_Array[KeyShape], _Array[IntTuple]]: ...
@overload
def rng_bit_generator(
    key: Any,
    shape: Any,
    dtype: DTypeLike = ...,
    algorithm: RandomAlgorithm = ...,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def rng_uniform[Shape: _Shape](
    a: _ArrayLike[[]],
    b: _ArrayLike[[]],
    shape: Shape,
) -> _Array[Shape]: ...
@overload
def rng_uniform(
    a: _ArrayLike[[]],
    b: _ArrayLike[[]],
    shape: Sequence[int] | int,
) -> _Array[IntTuple]: ...

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
@overload
def shape_as_value[Shape: _Shape](
    shape: Shape,
) -> _Array[shape_as_value_shape(Shape)]: ...
@overload
def shape_as_value(shape: Sequence[int]) -> _Array[IntTuple]: ...
def stage[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def stop_gradient[T](x: T) -> T: ...
def with_sharding_constraint[T](x: T, shardings: Any) -> T: ...
