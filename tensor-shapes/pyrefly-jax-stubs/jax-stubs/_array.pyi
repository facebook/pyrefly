# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The array class lives here rather than in `jax/__init__.pyi` so that
# `jax.numpy` can refer to it without importing its own parent package. Real
# JAX splits it out for the same reason, as `jax._src.basearray`.

from typing import Any, overload, Sequence

import shape_extensions
from jax._shapes import permute_shape, reduce_shape, reshape_shape, reverse_shape
from shape_extensions import broadcast, Flag, IntTuple, IntVar

type _Shape = IntTuple
type _AnyShape = tuple[Any, ...]
type _Axis = int | tuple[int, ...] | None
# The trailing `None` is not a legal argument to `reshape`. It is present because
# an `int | tuple[int, ...]` parameter cannot be iterated inside a DSL function
# after narrowing with `is_int_value` alone. See `reshape_shape`, which rejects it.
type _NewShape = int | tuple[int, ...] | None

@shape_extensions.shaped_array(shape="Shape")
class Array[Shape: _Shape = _AnyShape]:
    shape: Shape
    # JAX reverses every axis, at any rank, so this is not 2-D only.
    @property
    def T(self) -> Array[reverse_shape(Shape)]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> Any: ...
    @overload
    def __add__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __add__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __radd__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __radd__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __sub__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __sub__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __rsub__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __rsub__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __mul__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __mul__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __rmul__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __rmul__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __truediv__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __truediv__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __rtruediv__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __rtruediv__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __pow__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __pow__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __rpow__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __rpow__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    # Comparisons are elementwise and produce a boolean array, not a `bool`.
    # Without these, `a == b` falls through to `object.__eq__` and silently
    # infers `bool`, and `a > 0` is rejected outright.
    @overload
    def __eq__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __eq__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    # `object` keeps the override compatible with `object.__eq__`; JAX
    # compares elementwise against anything array-like.
    @overload
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __ne__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __ne__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    # `object` keeps the override compatible with `object.__ne__`; JAX
    # compares elementwise against anything array-like.
    @overload
    def __ne__(self, other: object) -> bool: ...
    @overload
    def __lt__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __lt__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __le__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __le__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __gt__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __gt__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __ge__(self, other: int | float | complex) -> Array[Shape]: ...
    @overload
    def __ge__[OtherShape: _Shape](
        self, other: Array[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __neg__(self) -> Array[Shape]: ...
    def __pos__(self) -> Array[Shape]: ...
    def __abs__(self) -> Array[Shape]: ...
    # Declared for 2-D operands only, which makes the operator stricter than
    # `jnp.matmul`: a batched `@` is reported as unsupported where the function
    # form is gradual. A gradual fallback overload here would also absorb the
    # mismatched-inner-dimension error, which is the most valuable check in
    # these stubs, so the narrower declaration is deliberate.
    @overload
    def __matmul__[N: IntVar, M: IntVar, P: IntVar](
        self: Array[[N, M]], other: Array[[M, P]]
    ) -> Array[[N, P]]: ...
    @overload
    def __matmul__[N: IntVar, M: IntVar](
        self: Array[[N, M]], other: Array[[M]]
    ) -> Array[[N]]: ...
    @overload
    def __matmul__[M: IntVar, P: IntVar](
        self: Array[[M]], other: Array[[M, P]]
    ) -> Array[[P]]: ...
    @overload
    def __matmul__[M: IntVar](self: Array[[M]], other: Array[[M]]) -> Array[[]]: ...
    @overload
    def transpose(self) -> Array[reverse_shape(Shape)]: ...
    @overload
    def transpose[Axes: Flag[_Axis]](
        self, axes: Axes, /
    ) -> Array[permute_shape(Shape, Axes)]: ...
    @overload
    def transpose(self, axes: Sequence[int], /) -> Array[IntTuple]: ...
    # JAX's variadic spelling, `a.transpose(1, 0)`, for the same reason as
    # `reshape` below: an argument list cannot be captured as a `Flag`.
    @overload
    def transpose(self, *axes: int) -> Array[IntTuple]: ...
    # Positional-only: JAX's signature is `reshape(self, *args, order=...)`, so
    # there is no `shape` keyword to bind.
    @overload
    def reshape[NewShape: Flag[_NewShape]](
        self, shape: NewShape, /, *, order: str = ..., out_sharding: Any = ...
    ) -> Array[reshape_shape(Shape, NewShape)]: ...
    # JAX's variadic spelling is accepted but intentionally not modeled: an
    # argument list cannot be captured as a `Flag`, so the shape is gradual and
    # `reshape_shape` never runs, which leaves the `-1` and negative-size checks
    # to runtime. Accepting it unvalidated is the intended behavior rather than a
    # gap to close: rejecting it would flag valid and very common JAX, and
    # modeling it needs a Pyrefly capability that does not exist. The tuple
    # spelling, `a.reshape((2, 6))`, is both exact and validated. Only the method
    # is variadic; `jnp.reshape(a, 2, 6)` is an error in JAX itself.
    @overload
    def reshape(
        self, shape: Sequence[int], /, *, order: str = ..., out_sharding: Any = ...
    ) -> Array[IntTuple]: ...
    @overload
    def reshape(
        self, *shape: int, order: str = ..., out_sharding: Any = ...
    ) -> Array[IntTuple]: ...
    @overload
    # Any non-tuple sequence axis is gradual; see `jax/numpy/__init__.pyi`.
    def sum[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
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
    def sum(
        self,
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
    def prod[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
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
    def prod(
        self,
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
    def mean[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
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
    def mean(
        self,
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
    def max[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
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
    def max(
        self,
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
    def min[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
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
    def min(
        self,
        axis: Sequence[int],
        *,
        keepdims: bool = False,
        dtype: Any = ...,
        out: Any = ...,
        initial: Any = ...,
        where: Any = ...,
        promote_integers: bool = ...,
    ) -> Array[IntTuple]: ...
