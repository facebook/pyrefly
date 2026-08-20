# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, overload, Sequence

from jax._array import Array as Array, Array as ndarray
from jax._shapes import (
    matmul_shape,
    permute_shape,
    reduce_shape,
    reshape_shape,
    reverse_shape,
)
from shape_extensions import broadcast, Flag, Int, IntTuple, IntVar

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None
# The trailing `None` is not a legal argument to `reshape`. It is present because
# an `int | tuple[int, ...]` parameter cannot be iterated inside a DSL function
# after narrowing with `is_int_value` alone. See `reshape_shape`, which rejects it.
type _NewShape = int | tuple[int, ...] | None

# Ranks 1 through 3 are exact; any other integer sequence, including a longer
# tuple or a list, falls through to a gradual overload rather than being
# rejected.
# TODO(stroxler): Replace these finite tuple-shape constructor overloads with a
# single `Shape: tuple[int, ...]` overload once carrier shapes flow through
# downstream shaped-array operations without degrading to unknown. The NumPy
# stubs carry the same limitation.
@overload
def zeros[N: IntVar](
    shape: Int[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def zeros[N: IntVar](
    shape: IntTuple[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def zeros[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M]]: ...
@overload
def zeros[N: IntVar, M: IntVar, K: IntVar](
    shape: IntTuple[N, M, K], dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M, K]]: ...
@overload
def zeros(
    shape: Sequence[int], dtype: Any = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def ones[N: IntVar](
    shape: Int[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def ones[N: IntVar](
    shape: IntTuple[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def ones[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M]]: ...
@overload
def ones[N: IntVar, M: IntVar, K: IntVar](
    shape: IntTuple[N, M, K], dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M, K]]: ...
@overload
def ones(
    shape: Sequence[int], dtype: Any = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def full[N: IntVar](
    shape: Int[N], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def full[N: IntVar](
    shape: IntTuple[N], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def full[N: IntVar, M: IntVar](
    shape: IntTuple[N, M], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M]]: ...
@overload
def full[N: IntVar, M: IntVar, K: IntVar](
    shape: IntTuple[N, M, K], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M, K]]: ...
@overload
def full(
    shape: Sequence[int], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[IntTuple]: ...

# `fill_value` stays `Any` because the rule JAX enforces is that it broadcasts
# *to* the requested shape, which is a constraint on the result rather than a
# computation of it. `broadcast(...)` computes a shape and cannot require that it
# equal the target, so `jnp.full((2, 3), jnp.ones(2))` is not rejected here. Using
# `broadcast(...)` anyway would trade this missed error for a wrong shape on
# `jnp.full((2, 3), jnp.ones((4, 2, 3)))`, which JAX also rejects.

# The single-argument form is the common one, so it carries its length exactly.
# The cost is that a negative literal infers a negative dimension where JAX
# returns an empty array: `jnp.arange(-3)` is `[-3]` here and `(0,)` at runtime.
# Clamping instead would cost the exact length on every ordinary call.
# TODO(stroxler): Represent an empty dimension, then clamp here. Pyrefly's shape
# domain currently excludes both negative and zero dimensions in a written
# annotation, so the correct answer for this call, `[0]`, is as unrepresentable
# as the wrong one. Note the ordering this implies: extending that check to
# inferred shapes before the clamp exists would turn today's wrong shape into a
# rejection of valid code. The NumPy stubs model `arange` the same way.
@overload
def arange[N: IntVar](
    start: Int[N], *, dtype: Any = ..., device: Any = ...
) -> Array[[N]]: ...

# A float `arange` is valid and its length is not an integer expression at all,
# so it is rank-1 with a gradual length.
@overload
def arange(start: float, *, dtype: Any = ..., device: Any = ...) -> Array[[int]]: ...

# The multi-argument forms mean `range(start, stop, step)`, whose length the DSL
# cannot compute. It needs a floor division and a clamp at zero: an
# `Int`-returning DSL function is restricted to an exact `Int +/- Flag[int]`, and
# an `Int` cannot be compared against a literal at all. Inferring `stop - start`
# without the clamp would claim a negative dimension for an empty range such as
# `jnp.arange(7, 2)`, which is worse than not knowing, so the result is rank-1
# with a gradual length.
# TODO(stroxler): Compute the length once the type-level DSL admits division and
# comparison in an `Int` return. The Torch migration is extending dimension
# arithmetic, so this should become expressible.
@overload
def arange(
    start: int | float,
    stop: int | float,
    step: int | float = ...,
    dtype: Any = ...,
) -> Array[[int]]: ...
@overload
def eye[N: IntVar](
    N: Int[N], M: None = ..., k: int = ..., dtype: Any = ..., *, device: Any = ...
) -> Array[[N, N]]: ...
@overload
def eye[N: IntVar, M: IntVar](
    N: Int[N], M: Int[M], k: int = ..., dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M]]: ...
def identity[N: IntVar](
    n: Int[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N, N]]: ...

# Shape-preserving elementwise unary functions.
def abs[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def negative[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def exp[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def log[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sqrt[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def square[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sin[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def cos[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def tanh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...

# Broadcasting elementwise binary functions. Each takes a scalar in either
# position as well as an array: rejecting `jnp.add(a, 1)` would flag valid code.
@overload
def add[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def add[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def add[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def subtract[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def subtract[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def subtract[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def multiply[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def multiply[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def multiply[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def divide[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def divide[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def divide[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def maximum[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def maximum[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def maximum[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def minimum[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def minimum[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def minimum[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...

# This MVP models only two-dimensional operands, matching the NumPy stubs.
def matmul[LeftShape: _Shape, RightShape: _Shape](
    a: Array[LeftShape], b: Array[RightShape]
) -> Array[matmul_shape(LeftShape, RightShape)]: ...
@overload
def transpose[Shape: _Shape](
    a: Array[Shape], axes: None = None
) -> Array[reverse_shape(Shape)]: ...
@overload
def transpose[Shape: _Shape, Axes: Flag[_Axis]](
    a: Array[Shape], axes: Axes = None
) -> Array[permute_shape(Shape, Axes)]: ...
@overload
def transpose[Shape: _Shape](
    a: Array[Shape], axes: Sequence[int]
) -> Array[IntTuple]: ...

# A single int or tuple, matching JAX: the free function is not variadic, so
# `jnp.reshape(a, 2, 3)` is an error there. `Array.reshape` is the variadic one.
@overload
def reshape[Shape: _Shape, NewShape: Flag[_NewShape]](
    a: Array[Shape],
    shape: NewShape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[reshape_shape(Shape, NewShape)]: ...
@overload
def reshape(
    a: Array[Any],
    shape: Sequence[int],
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...

# JAX accepts any integer sequence for an axis, but only a tuple is a Flag
# domain, so any other sequence yields a gradual shape. Rejecting it would flag
# valid code. The exact overload is declared first so that a tuple resolves to
# it rather than being absorbed by the fallback.
@overload
def sum[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def sum[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def prod[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def prod[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def mean[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def mean[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def max[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def max[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def min[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def min[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: Any = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...

float32: Any
float64: Any
int32: Any
int64: Any
bool_: Any
