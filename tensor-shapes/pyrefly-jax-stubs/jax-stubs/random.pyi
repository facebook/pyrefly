# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, overload

from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._src.sharding_impls import (
    NamedSharding as _NamedSharding,
    PartitionSpec as _PartitionSpec,
)
from jax.typing import DTypeLike
from shape_extensions import Int, IntTuple, IntVar

type _Shape = IntTuple

# Keep public APIs gradual until a later stack commit gives them precise shapes.
def __getattr__(name: str) -> Any: ...
def key(
    seed: int | _ArrayLike[[]], *, impl: Any | None = None, dtype: Any | None = None
) -> _Array[[]]: ...
def PRNGKey(
    seed: int | _ArrayLike[[]], *, impl: Any | None = None
) -> _Array[IntTuple]: ...
def fold_in[KeyShape: _Shape = []](
    key: _ArrayLike[KeyShape], data: int | _ArrayLike[[]]
) -> _Array[KeyShape]: ...
@overload
def split(key: _ArrayLike[[]]) -> _Array[[2]]: ...
@overload
def split[N: IntVar](key: _ArrayLike[[]], num: Int[N]) -> _Array[[N]]: ...
@overload
def split[NumShape: _Shape](key: _ArrayLike[[]], num: NumShape) -> _Array[NumShape]: ...
@overload
def split(
    key: _ArrayLike[IntTuple], num: int | tuple[int, ...] = 2
) -> _Array[IntTuple]: ...
@overload
def bits[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def bits(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def uniform[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    minval: _ArrayLike[IntTuple] = 0.0,
    maxval: _ArrayLike[IntTuple] = 1.0,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def uniform(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    minval: _ArrayLike[IntTuple] = 0.0,
    maxval: _ArrayLike[IntTuple] = 1.0,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def randint[Shape: _Shape](
    key: _ArrayLike[IntTuple],
    shape: Shape,
    minval: _ArrayLike[IntTuple],
    maxval: _ArrayLike[IntTuple],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def randint(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    minval: _ArrayLike[IntTuple],
    maxval: _ArrayLike[IntTuple],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def normal[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def normal(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
