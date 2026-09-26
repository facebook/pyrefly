# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, overload

from jax._array import Array as _Array, ArrayLike as _ArrayLike
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
