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
from shape_extensions import broadcast, Int, IntTuple, IntVar

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
@overload
def cauchy[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def cauchy(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def exponential[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def exponential(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def laplace[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def laplace(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def logistic[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def logistic(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def gumbel[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    mode: str | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def gumbel(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    mode: str | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def maxwell[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def maxwell(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def rademacher[Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[Shape]: ...
@overload
def rademacher(
    key: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def bernoulli[PShape: _Shape = []](
    key: _ArrayLike[IntTuple],
    p: _ArrayLike[PShape] = 0.5,
    shape: None = None,
    mode: str = "low",
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[PShape]: ...
@overload
def bernoulli[PShape: _Shape = [], Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    p: _ArrayLike[PShape] = 0.5,
    shape: Shape = (),
    mode: str = "low",
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(PShape, Shape)]: ...
@overload
def bernoulli(
    key: _ArrayLike[IntTuple],
    p: _ArrayLike[IntTuple] = 0.5,
    shape: Sequence[int] = (),
    mode: str = "low",
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def gamma[ParameterShape: _Shape](
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[ParameterShape],
    shape: None = None,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[ParameterShape]: ...
@overload
def gamma[ParameterShape: _Shape, Shape: _Shape](
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[ParameterShape],
    shape: Shape,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(ParameterShape, Shape)]: ...
@overload
def gamma(
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def poisson[ParameterShape: _Shape](
    key: _ArrayLike[IntTuple],
    lam: _ArrayLike[ParameterShape],
    shape: None = None,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[ParameterShape]: ...
@overload
def poisson[ParameterShape: _Shape, Shape: _Shape](
    key: _ArrayLike[IntTuple],
    lam: _ArrayLike[ParameterShape],
    shape: Shape,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(ParameterShape, Shape)]: ...
@overload
def poisson(
    key: _ArrayLike[IntTuple],
    lam: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def beta[AShape: _Shape = [], BShape: _Shape = []](
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[AShape],
    b: _ArrayLike[BShape],
    shape: None = None,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(AShape, BShape)]: ...
@overload
def beta[AShape: _Shape = [], BShape: _Shape = [], Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[AShape],
    b: _ArrayLike[BShape],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(broadcast(AShape, BShape), Shape)]: ...
@overload
def beta(
    key: _ArrayLike[IntTuple],
    a: _ArrayLike[IntTuple],
    b: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def f[NumShape: _Shape = [], DenShape: _Shape = []](
    key: _ArrayLike[IntTuple],
    dfnum: _ArrayLike[NumShape],
    dfden: _ArrayLike[DenShape],
    shape: None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(NumShape, DenShape)]: ...
@overload
def f[NumShape: _Shape = [], DenShape: _Shape = [], Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    dfnum: _ArrayLike[NumShape],
    dfden: _ArrayLike[DenShape],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[broadcast(broadcast(NumShape, DenShape), Shape)]: ...
@overload
def f(
    key: _ArrayLike[IntTuple],
    dfnum: _ArrayLike[IntTuple],
    dfden: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def binomial[NShape: _Shape = [], PShape: _Shape = []](
    key: _ArrayLike[IntTuple],
    n: _ArrayLike[NShape],
    p: _ArrayLike[PShape],
    shape: None = None,
    dtype: DTypeLike | None = None,
) -> _Array[broadcast(NShape, PShape)]: ...
@overload
def binomial[NShape: _Shape = [], PShape: _Shape = [], Shape: _Shape = []](
    key: _ArrayLike[IntTuple],
    n: _ArrayLike[NShape],
    p: _ArrayLike[PShape],
    shape: Shape = (),
    dtype: DTypeLike | None = None,
) -> _Array[broadcast(broadcast(NShape, PShape), Shape)]: ...
@overload
def binomial(
    key: _ArrayLike[IntTuple],
    n: _ArrayLike[IntTuple],
    p: _ArrayLike[IntTuple],
    shape: Sequence[int],
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
