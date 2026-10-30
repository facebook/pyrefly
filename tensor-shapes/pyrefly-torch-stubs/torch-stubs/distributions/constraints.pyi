# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.constraints."""

from collections.abc import Sequence
from typing import Any, Callable, Literal, Never, overload, TypeIs

from shape_extensions import broadcast, IntTuple, IntVar
from torch import Tensor

class Constraint:
    """Base class for constraints."""

    is_discrete: bool = False
    event_dim: int = 0

    def check(self, value: Tensor) -> Tensor: ...

class _Boolean(Constraint):
    is_discrete: bool = True

    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[S]: ...

class _OneHot(Constraint):
    is_discrete: bool = True
    event_dim: int = 1

    def check[S: IntTuple, N: IntVar](self, value: Tensor[[*S, N]]) -> Tensor[S]: ...

class _Simplex(Constraint):
    event_dim: int = 1

    def check[S: IntTuple, N: IntVar](self, value: Tensor[[*S, N]]) -> Tensor[S]: ...

class _Square(Constraint):
    event_dim: int = 2

    def check[S: IntTuple, M: IntVar, N: IntVar](
        self, value: Tensor[[*S, M, N]]
    ) -> Tensor[S]: ...

class _Symmetric(_Square): ...

class _LowerTriangular(Constraint):
    event_dim: int = 2

    def check[S: IntTuple, M: IntVar, N: IntVar](
        self, value: Tensor[[*S, M, N]]
    ) -> Tensor[S]: ...

class _LowerCholesky(Constraint):
    event_dim: int = 2

    def check[S: IntTuple, M: IntVar, N: IntVar](
        self, value: Tensor[[*S, M, N]]
    ) -> Tensor[S]: ...

class _CorrCholesky(Constraint):
    event_dim: int = 2

    def check[S: IntTuple, M: IntVar, N: IntVar](
        self, value: Tensor[[*S, M, N]]
    ) -> Tensor[S]: ...

class _PositiveSemidefinite(_Symmetric): ...
class _PositiveDefinite(_Symmetric): ...

class _IntegerGreaterThan(Constraint):
    is_discrete: bool = True
    lower_bound: int

    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[S]: ...

class _GreaterThan[Bounds: IntTuple = []](Constraint):
    lower_bound: float | Tensor[Bounds]

    def __init__(self, lower_bound: float | Tensor[Bounds]) -> None: ...
    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[broadcast(S, Bounds)]: ...

class _GreaterThanEq[Bounds: IntTuple = []](Constraint):
    lower_bound: float | Tensor[Bounds]

    def __init__(self, lower_bound: float | Tensor[Bounds]) -> None: ...
    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[broadcast(S, Bounds)]: ...

class _LessThan[Bounds: IntTuple = []](Constraint):
    upper_bound: float | Tensor[Bounds]

    def __init__(self, upper_bound: float | Tensor[Bounds]) -> None: ...
    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[broadcast(S, Bounds)]: ...

class _Interval(Constraint):
    lower_bound: float
    upper_bound: float

    def check[S: IntTuple](self, value: Tensor[S]) -> Tensor[S]: ...

class _IntegerInterval[Lower: IntTuple = [], Upper: IntTuple = []](Constraint):
    is_discrete: bool = True
    lower_bound: int | Tensor[Lower]
    upper_bound: int | Tensor[Upper]

    def __init__(
        self, lower_bound: int | Tensor[Lower], upper_bound: int | Tensor[Upper]
    ) -> None: ...
    def check[S: IntTuple](
        self, value: Tensor[S]
    ) -> Tensor[broadcast(broadcast(S, Lower), Upper)]: ...

class _HalfOpenInterval[Lower: IntTuple = [], Upper: IntTuple = []](Constraint):
    lower_bound: float | Tensor[Lower]
    upper_bound: float | Tensor[Upper]

    def __init__(
        self, lower_bound: float | Tensor[Lower], upper_bound: float | Tensor[Upper]
    ) -> None: ...
    def check[S: IntTuple](
        self, value: Tensor[S]
    ) -> Tensor[broadcast(broadcast(S, Lower), Upper)]: ...

class _Multinomial[Bounds: IntTuple = []](Constraint):
    is_discrete: bool = True
    event_dim: int = 1
    upper_bound: int | Tensor[Bounds]

    def __init__(self, upper_bound: int | Tensor[Bounds]) -> None: ...
    def check[S: IntTuple, N: IntVar](
        self, value: Tensor[[*S, N]]
    ) -> Tensor[broadcast(S, Bounds)]: ...

class _Cat(Constraint):
    """Apply constraints to adjacent slices along a tensor dimension."""

    cseq: list[Constraint]
    lengths: list[int]
    dim: int

    def __init__(
        self,
        cseq: Sequence[Constraint],
        dim: int = 0,
        lengths: Sequence[int] | None = None,
    ) -> None: ...

class _Stack(Constraint):
    """Apply constraints to the slices of a stacked tensor."""

    cseq: list[Constraint]
    dim: int

    def __init__(self, cseq: Sequence[Constraint], dim: int = 0) -> None: ...

class _IndependentConstraint[EventDim: int = int](Constraint):
    """Aggregate a base constraint over its trailing event dimensions."""

    base_constraint: Constraint
    reinterpreted_batch_ndims: int

    def __init__(
        self, base_constraint: Constraint, reinterpreted_batch_ndims: int
    ) -> None: ...
    @overload
    def check[S: IntTuple, N: IntVar](
        self: _IndependentConstraint[Literal[1]], value: Tensor[[*S, N]]
    ) -> Tensor[S]: ...
    @overload
    def check(self, value: Tensor) -> Tensor: ...

class MixtureSameFamilyConstraint(Constraint):
    """Check a value against every component constraint in a mixture."""

    base_constraint: Constraint

    def __init__(self, base_constraint: Constraint) -> None: ...

class _Dependent(Constraint):
    """Mark support that cannot be checked without other distribution values."""

    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def __init__(self, *, is_discrete: bool = ..., event_dim: int = ...) -> None: ...
    def __call__(
        self, *, is_discrete: bool = ..., event_dim: int = ...
    ) -> _Dependent: ...
    def check(self, value: Tensor) -> Never: ...

class _DependentProperty(property, _Dependent):
    """Expose dependent support on a class and a property on its instances."""

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        is_discrete: bool | None = ...,
        event_dim: int | None = ...,
    ) -> None: ...
    def __call__(self, fn: Callable[..., Any]) -> _DependentProperty: ...

real: Constraint
boolean: _Boolean
one_hot: _OneHot
simplex: _Simplex
square: _Square
symmetric: _Symmetric
lower_triangular: _LowerTriangular
lower_cholesky: _LowerCholesky
corr_cholesky: _CorrCholesky
positive_semidefinite: _PositiveSemidefinite
positive_definite: _PositiveDefinite
positive: _GreaterThan[[]]
nonnegative: _GreaterThanEq[[]]
positive_integer: _IntegerGreaterThan
nonnegative_integer: _IntegerGreaterThan
unit_interval: _Interval
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
integer_interval = _IntegerInterval
half_open_interval = _HalfOpenInterval
multinomial = _Multinomial
cat = _Cat
stack = _Stack
independent = _IndependentConstraint
real_vector: _IndependentConstraint[Literal[1]]
dependent: _Dependent
dependent_property = _DependentProperty

def interval(lower_bound: float, upper_bound: float) -> Constraint: ...
def is_dependent(constraint: Constraint) -> TypeIs[_Dependent]: ...
