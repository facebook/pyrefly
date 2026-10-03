# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.constraints."""

from typing import Any

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

def interval(lower_bound: float, upper_bound: float) -> Constraint: ...

# TODO: Replace these availability stubs with precise declarations.
MixtureSameFamilyConstraint: Any
cat: Any
dependent: Any
dependent_property: Any
half_open_interval: Any
independent: Any
integer_interval: Any
is_dependent: Any
multinomial: Any
real_vector: Any
stack: Any
