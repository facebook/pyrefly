# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Literal, overload, Sequence

from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._shapes import glu_shape, one_hot_shape, reduce_shape
from shape_extensions import broadcast, Flag, Int, IntTuple, IntVar

from . import initializers as initializers

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None

# Every activation JAX exposes here is elementwise, so the shape is preserved.
# `softmax`, `log_softmax`, and `standardize` normalize along `axis` but do not reduce it away.
# `elu`, `leaky_relu`, `celu`, and `squareplus` take their parameter as an array as well as a scalar,
# and an array parameter broadcasts against the input rather than preserving its shape.
# `glu` splits the specified axis in half.
# `logsumexp` and `logmeanexp` perform reductions over the specified axis.
# `one_hot` inserts a new dimension with `num_classes` at the specified axis.
def relu[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def relu6[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def sigmoid[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def softplus[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def sparse_plus[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def sparse_sigmoid[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def soft_sign[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def silu[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def swish[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def log_sigmoid[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def hard_sigmoid[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def hard_silu[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def hard_swish[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def hard_tanh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def tanh[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def selu[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def mish[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def identity[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def log1mexp[Shape: _Shape = []](x: _ArrayLike[Shape]) -> _Array[Shape]: ...
def elu[Shape: _Shape = [], ParamShape: _Shape = []](
    x: _ArrayLike[Shape], alpha: _ArrayLike[ParamShape] = ...
) -> _Array[broadcast(Shape, ParamShape)]: ...
def leaky_relu[Shape: _Shape = [], ParamShape: _Shape = []](
    x: _ArrayLike[Shape], negative_slope: _ArrayLike[ParamShape] = ...
) -> _Array[broadcast(Shape, ParamShape)]: ...
def celu[Shape: _Shape = [], ParamShape: _Shape = []](
    x: _ArrayLike[Shape], alpha: _ArrayLike[ParamShape] = ...
) -> _Array[broadcast(Shape, ParamShape)]: ...
def squareplus[Shape: _Shape = [], ParamShape: _Shape = []](
    x: _ArrayLike[Shape], b: _ArrayLike[ParamShape] = ...
) -> _Array[broadcast(Shape, ParamShape)]: ...
def gelu[Shape: _Shape = []](
    x: _ArrayLike[Shape], approximate: bool = ...
) -> _Array[Shape]: ...
@overload
def glu[Shape: _Shape = [], Axis: Flag[int] = -1](
    x: _ArrayLike[Shape], axis: Axis = -1
) -> _Array[glu_shape(Shape, Axis)]: ...
@overload
def glu[Shape: _Shape = []](
    x: _ArrayLike[Shape], axis: int = -1
) -> _Array[IntTuple]: ...
def softmax[Shape: _Shape = []](
    x: _ArrayLike[Shape], axis: Sequence[int] | int | None = ..., where: Any = ...
) -> _Array[Shape]: ...
def log_softmax[Shape: _Shape = []](
    x: _ArrayLike[Shape], axis: Sequence[int] | int | None = ..., where: Any = ...
) -> _Array[Shape]: ...
def standardize[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axis: Sequence[int] | int | None = -1,
    mean: _ArrayLike[Any] | None = None,
    variance: _ArrayLike[Any] | None = None,
    epsilon: float = 1e-5,
    where: Any = None,
) -> _Array[Shape]: ...
@overload
def logsumexp[
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
    Shape: _Shape = [],
](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    b: _ArrayLike[Any] | None = None,
    keepdims: KeepDims = False,
    return_sign: Literal[False] = False,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def logsumexp[
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
    Shape: _Shape = [],
](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    b: _ArrayLike[Any] | None = None,
    keepdims: KeepDims = False,
    return_sign: Literal[True] = ...,
    where: Any = None,
) -> tuple[
    _Array[reduce_shape(Shape, Axis, KeepDims)],
    _Array[reduce_shape(Shape, Axis, KeepDims)],
]: ...
@overload
def logsumexp[
    Shape: _Shape = [],
](
    a: _ArrayLike[Shape],
    axis: Sequence[int] = ...,
    b: _ArrayLike[Any] | None = None,
    keepdims: bool = False,
    return_sign: Literal[False] = False,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def logsumexp[
    Shape: _Shape = [],
](
    a: _ArrayLike[Shape],
    axis: Sequence[int] = ...,
    b: _ArrayLike[Any] | None = None,
    keepdims: bool = False,
    return_sign: Literal[True] = ...,
    where: Any = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def logsumexp(
    a: _ArrayLike[Any],
    axis: Sequence[int] | int | None = None,
    b: _ArrayLike[Any] | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: Any = None,
) -> _Array[IntTuple] | tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def logmeanexp[
    Axis: Flag[_Axis],
    KeepDims: Flag[bool],
    Shape: _Shape = [],
](
    x: _ArrayLike[Shape],
    axis: Axis = None,
    where: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def logmeanexp[
    Shape: _Shape = [],
](
    x: _ArrayLike[Shape],
    axis: Sequence[int] = ...,
    where: Any = None,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...
@overload
def logmeanexp(
    x: _ArrayLike[Any],
    axis: Sequence[int] | int | None = None,
    where: Any = None,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...
@overload
def one_hot[
    Shape: _Shape = [],
    NumClasses: IntVar = 0,
    Axis: Flag[int] = -1,
](
    x: _ArrayLike[Shape],
    num_classes: Int[NumClasses],
    *,
    dtype: Any = None,
    axis: Axis = -1,
    out_sharding: Any = None,
) -> _Array[one_hot_shape(Shape, Int[NumClasses], Axis)]: ...
@overload
def one_hot[
    Shape: _Shape = [],
](
    x: _ArrayLike[Shape],
    num_classes: int,
    *,
    dtype: Any = None,
    axis: int = -1,
    out_sharding: Any = None,
) -> _Array[IntTuple]: ...
@overload
def dot_product_attention[
    QueryShape: _Shape = [],
](
    query: _ArrayLike[QueryShape],
    key: _ArrayLike[Any],
    value: _ArrayLike[Any],
    bias: _ArrayLike[Any] | None = None,
    mask: _ArrayLike[Any] | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: _ArrayLike[Any] | None = None,
    key_value_seq_lengths: _ArrayLike[Any] | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    return_residual: Literal[False] = False,
    implementation: str | None = None,
) -> _Array[QueryShape]: ...
@overload
def dot_product_attention[
    QueryShape: _Shape = [],
](
    query: _ArrayLike[QueryShape],
    key: _ArrayLike[Any],
    value: _ArrayLike[Any],
    bias: _ArrayLike[Any] | None = None,
    mask: _ArrayLike[Any] | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: _ArrayLike[Any] | None = None,
    key_value_seq_lengths: _ArrayLike[Any] | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    return_residual: Literal[True] = ...,
    implementation: str | None = None,
) -> tuple[_Array[QueryShape], _Array[IntTuple]]: ...
@overload
def dot_product_attention(
    query: _ArrayLike[Any],
    key: _ArrayLike[Any],
    value: _ArrayLike[Any],
    bias: _ArrayLike[Any] | None = None,
    mask: _ArrayLike[Any] | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: _ArrayLike[Any] | None = None,
    key_value_seq_lengths: _ArrayLike[Any] | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    return_residual: bool = False,
    implementation: str | None = None,
) -> _Array[IntTuple] | tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def scaled_matmul[
    Batch: IntTuple,
    M: IntVar,
    N: IntVar,
    K: IntVar,
](
    lhs: _ArrayLike[[*Batch, M, K]],
    rhs: _ArrayLike[[*Batch, N, K]],
    lhs_scales: _ArrayLike[Any],
    rhs_scales: _ArrayLike[Any],
    preferred_element_type: Any = None,
) -> _Array[[*Batch, M, N]]: ...
@overload
def scaled_matmul(
    lhs: _ArrayLike[Any],
    rhs: _ArrayLike[Any],
    lhs_scales: _ArrayLike[Any],
    rhs_scales: _ArrayLike[Any],
    preferred_element_type: Any = None,
) -> _Array[IntTuple]: ...
def scaled_dot_general(
    lhs: _ArrayLike[Any],
    rhs: _ArrayLike[Any],
    dimension_numbers: Any,
    lhs_scale: _ArrayLike[Any] | None = None,
    rhs_scale: _ArrayLike[Any] | None = None,
    **kwargs: Any,
) -> _Array[IntTuple]: ...
def get_scaled_dot_general_config(mode: str, **kwargs: Any) -> Any: ...
