# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The array class lives here rather than in `jax/__init__.pyi` so that
# `jax.numpy` can refer to it without importing its own parent package. Real
# JAX splits it out for the same reason, as `jax._src.basearray`.

from types import EllipsisType
from typing import Any, overload, Protocol, Sequence, SupportsIndex

import numpy as np
from jax._shapes import (
    compress_shape,
    diagonal_shape,
    dot_shape,
    matmul_shape,
    permute_shape,
    ravel_shape,
    reduce_shape,
    repeat_shape,
    reshape_shape,
    reverse_shape,
    sort_shape,
    squeeze_shape,
    swapaxes_shape,
    take_shape,
    trace_shape,
)
from jax._src.sharding_impls import (
    NamedSharding as _NamedSharding,
    PartitionSpec as _PartitionSpec,
)
from jax.typing import DTypeLike
from shape_extensions import broadcast, Flag, Index, index_shape, Int, IntTuple, IntVar

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None
type _Scalar = bool | int | float | complex | np.number
# Note: when using _ArrayLike in an annotation, the Shape passed to it must
# have a default value of [] for scalars to be handled properly.
type ArrayLike[Shape: _Shape] = Array[Shape] | np.ndarray[Shape] | _Scalar

class _ArrayIndex(Protocol):
    @property
    def shape(self) -> object: ...
    @property
    def dtype(self) -> object: ...

type _IntegerSequence = (
    # Lists are intentionally accepted through contextual typing of literals, not as a general
    # sequence abstraction. This avoids requiring runtime code to replace existing list syntax.
    list[SupportsIndex | _IntegerSequence]
    | tuple[SupportsIndex | _IntegerSequence, ...]
)
type _BasicIndex = SupportsIndex | slice | _IntegerSequence | None | EllipsisType
# The trailing `None` is not a legal argument to `reshape`. It is present because
# an `int | tuple[int, ...]` parameter cannot be iterated inside a DSL function
# after narrowing with `is_int_value` alone. See `reshape_shape`, which rejects it.
type _NewShape = int | tuple[int, ...] | None

class Array[Shape: _Shape = _Shape]:
    shape: Shape
    @overload
    def __getitem__[I: Index](self, index: I) -> Array[index_shape(Shape, I)]: ...
    @overload
    # JAX accepts array operands. The structural, tuple, and list-literal arms also provide
    # gradual compatibility for values that JAX may reject at runtime.
    def __getitem__(
        self,
        index: _BasicIndex | _ArrayIndex | tuple[_BasicIndex | _ArrayIndex, ...],
    ) -> Array[IntTuple]: ...
    # JAX reverses every axis, at any rank, so this is not 2-D only.
    @property
    def T(self) -> Array[reverse_shape(Shape)]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> Any: ...
    def __add__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __radd__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __sub__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __rsub__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __mul__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __rmul__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __truediv__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __rtruediv__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __pow__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __rpow__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __eq__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __ne__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    @overload
    def __ne__(self, other: object) -> bool: ...
    def __lt__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __le__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __gt__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __ge__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[broadcast(Shape, OtherShape)]: ...
    def __neg__(self) -> Array[Shape]: ...
    def __pos__(self) -> Array[Shape]: ...
    def __abs__(self) -> Array[Shape]: ...
    def __matmul__[OtherShape: _Shape = []](
        self, other: ArrayLike[OtherShape]
    ) -> Array[matmul_shape(Shape, OtherShape)]: ...
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
        self,
        shape: NewShape,
        /,
        *,
        order: str = ...,
        out_sharding: _NamedSharding | _PartitionSpec | None = ...,
    ) -> Array[reshape_shape(Shape, NewShape)]: ...
    @overload
    def reshape[NewShape: IntTuple](self, *shape: *NewShape) -> Array[NewShape]: ...
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
        self,
        shape: Sequence[int],
        /,
        *,
        order: str = ...,
        out_sharding: _NamedSharding | _PartitionSpec | None = ...,
    ) -> Array[IntTuple]: ...
    @overload
    def reshape(
        self,
        *shape: int,
        order: str = ...,
        out_sharding: _NamedSharding | _PartitionSpec | None = ...,
    ) -> Array[IntTuple]: ...
    def ravel(self, order: str = "C") -> Array[ravel_shape(Shape)]: ...
    @overload
    def squeeze[Axis: Flag[_Axis] = None](
        self, axis: Axis = None
    ) -> Array[squeeze_shape(Shape, Axis)]: ...
    @overload
    def squeeze(self, axis: Sequence[int] | None = None) -> Array[IntTuple]: ...
    @overload
    def swapaxes[Axis1: Flag[int], Axis2: Flag[int]](
        self, axis1: Axis1, axis2: Axis2
    ) -> Array[swapaxes_shape(Shape, Axis1, Axis2)]: ...
    @overload
    def swapaxes(self, axis1: int, axis2: int) -> Array[IntTuple]: ...
    @overload
    def repeat[Repeats: Int, Axis: Flag[int | None]](
        self,
        repeats: Repeats,
        axis: Axis = None,
        *,
        total_repeat_length: None = None,
    ) -> Array[repeat_shape(Shape, Repeats, Axis)]: ...
    @overload
    def repeat(
        self,
        repeats: Array[Any] | int | Sequence[int],
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
    ) -> Array[IntTuple]: ...
    @overload
    # Any non-tuple sequence axis is gradual; see `jax/numpy/__init__.pyi`.
    def sum[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        *,
        keepdims: KeepDims = False,
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
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
        dtype: DTypeLike | None = ...,
        out: Any = ...,
        initial: Any = ...,
        where: Any = ...,
        promote_integers: bool = ...,
    ) -> Array[IntTuple]: ...
    @overload
    def all[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        out: Any = None,
        keepdims: KeepDims = False,
        *,
        where: Any = None,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def all(
        self,
        axis: Sequence[int],
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def any[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        out: Any = None,
        keepdims: KeepDims = False,
        *,
        where: Any = None,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def any(
        self,
        axis: Sequence[int],
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def std[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 0,
        keepdims: KeepDims = False,
        *,
        where: Any = None,
        mean: Any = None,
        correction: Any = None,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def std(
        self,
        axis: Sequence[int],
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: Any = None,
        mean: Any = None,
        correction: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def var[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 0,
        keepdims: KeepDims = False,
        *,
        where: Any = None,
        mean: Any = None,
        correction: Any = None,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def var(
        self,
        axis: Sequence[int],
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: Any = None,
        mean: Any = None,
        correction: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def ptp[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        out: Any = None,
        keepdims: KeepDims = False,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def ptp(
        self,
        axis: Sequence[int],
        out: Any = None,
        keepdims: bool = False,
    ) -> Array[IntTuple]: ...
    @overload
    def argmax[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        out: Any = None,
        keepdims: KeepDims = False,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def argmax(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool | None = None,
    ) -> Array[IntTuple]: ...
    @overload
    def argmin[Axis: Flag[_Axis], KeepDims: Flag[bool]](
        self,
        axis: Axis = None,
        out: Any = None,
        keepdims: KeepDims = False,
    ) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
    @overload
    def argmin(
        self,
        axis: int | None = None,
        out: Any = None,
        keepdims: bool | None = None,
    ) -> Array[IntTuple]: ...
    @overload
    def cumsum(
        self,
        axis: int,
        dtype: DTypeLike | None = None,
        out: Any = None,
    ) -> Array[Shape]: ...
    @overload
    def cumsum(
        self,
        axis: None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def cumprod(
        self,
        axis: int,
        dtype: DTypeLike | None = None,
        out: Any = None,
    ) -> Array[Shape]: ...
    @overload
    def cumprod(
        self,
        axis: None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
    ) -> Array[IntTuple]: ...
    def dot[OtherShape: _Shape = []](
        self,
        b: ArrayLike[OtherShape],
        *,
        precision: Any = None,
        preferred_element_type: Any = None,
        out_sharding: _NamedSharding | _PartitionSpec | None = None,
    ) -> Array[dot_shape(Shape, OtherShape)]: ...
    @overload
    def diagonal[
        Offset: Flag[int] = 0,
        Axis1: Flag[int] = 0,
        Axis2: Flag[int] = 1,
    ](
        self,
        offset: Offset = 0,
        axis1: Axis1 = 0,
        axis2: Axis2 = 1,
    ) -> Array[diagonal_shape(Shape, Offset, Axis1, Axis2)]: ...
    @overload
    def diagonal(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> Array[IntTuple]: ...
    @overload
    def trace[
        Offset: Flag[int] = 0,
        Axis1: Flag[int] = 0,
        Axis2: Flag[int] = 1,
    ](
        self,
        offset: Offset = 0,
        axis1: Axis1 = 0,
        axis2: Axis2 = 1,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array[trace_shape(Shape, Offset, Axis1, Axis2)]: ...
    @overload
    def trace(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array[IntTuple]: ...
    @overload
    def sort[Axis: Flag[int | None] = -1](
        self,
        axis: Axis = -1,
        *,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False,
    ) -> Array[sort_shape(Shape, Axis)]: ...
    @overload
    def sort(
        self,
        axis: int | None = -1,
        *,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False,
    ) -> Array[IntTuple]: ...
    @overload
    def argsort[Axis: Flag[int | None] = -1](
        self,
        axis: Axis = -1,
        *,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False,
    ) -> Array[sort_shape(Shape, Axis)]: ...
    @overload
    def argsort(
        self,
        axis: int | None = -1,
        *,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False,
    ) -> Array[IntTuple]: ...
    @overload
    def argpartition[Axis: Flag[int] = -1](
        self,
        kth: int | Sequence[int],
        axis: Axis = -1,
    ) -> Array[sort_shape(Shape, Axis)]: ...
    @overload
    def argpartition(
        self,
        kth: int | Sequence[int],
        axis: int = -1,
    ) -> Array[IntTuple]: ...
    @overload
    def nonzero[Size: IntVar](
        self,
        *,
        fill_value: Any = None,
        size: Int[Size],
    ) -> tuple[Array[[Size]], ...]: ...
    @overload
    def nonzero(
        self,
        *,
        fill_value: Any = None,
        size: int | None = None,
    ) -> tuple[Array[IntTuple], ...]: ...
    @overload
    def searchsorted[OtherShape: _Shape = []](
        self,
        v: ArrayLike[OtherShape],
        side: str = "left",
        sorter: Any = None,
        *,
        method: str = "scan",
    ) -> Array[OtherShape]: ...
    @overload
    def searchsorted(
        self,
        v: Any,
        side: str = "left",
        sorter: Any = None,
        *,
        method: str = "scan",
    ) -> Array[IntTuple]: ...
    @overload
    def choose(
        self,
        choices: Sequence[Array[Shape] | _Scalar],
        out: Any = None,
        mode: str = "raise",
    ) -> Array[Shape]: ...
    @overload
    def choose(
        self,
        choices: Any,
        out: Any = None,
        mode: str = "raise",
    ) -> Array[IntTuple]: ...
    @overload
    def clip[
        MinShape: _Shape = [],
        MaxShape: _Shape = [],
    ](
        self,
        min: ArrayLike[MinShape] | None = None,
        max: ArrayLike[MaxShape] | None = None,
    ) -> Array[broadcast(broadcast(Shape, MinShape), MaxShape)]: ...
    @overload
    def clip(
        self,
        min: Any = None,
        max: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def take[IdxShape: _Shape = [], Axis: Flag[int | None] = None](
        self,
        indices: Array[IdxShape] | np.ndarray[IdxShape] | int | np.integer,
        axis: Axis = None,
        out: None = None,
        mode: str | None = None,
        unique_indices: bool = False,
        indices_are_sorted: bool = False,
        fill_value: Any = None,
    ) -> Array[take_shape(Shape, IdxShape, Axis)]: ...
    @overload
    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: None = None,
        mode: str | None = None,
        unique_indices: bool = False,
        indices_are_sorted: bool = False,
        fill_value: Any = None,
    ) -> Array[IntTuple]: ...
    @overload
    def compress[Size: Flag[int], Axis: Flag[int | None] = None](
        self,
        condition: Any,
        axis: Axis = None,
        out: None = None,
        *,
        size: Size,
        fill_value: Any = 0,
    ) -> Array[compress_shape(Shape, Size, Axis)]: ...
    @overload
    def compress(
        self,
        condition: Any,
        axis: int | None = None,
        out: None = None,
        *,
        size: int | None = None,
        fill_value: Any = 0,
    ) -> Array[IntTuple]: ...
