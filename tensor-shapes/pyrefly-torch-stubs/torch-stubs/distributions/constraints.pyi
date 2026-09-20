# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.constraints."""

from typing import Any

class Constraint:
    """Base class for constraints."""

    ...

real: Constraint

def interval(lower_bound: float, upper_bound: float) -> Constraint: ...

# TODO: Replace these availability stubs with precise declarations.
MixtureSameFamilyConstraint: Any
boolean: Any
cat: Any
corr_cholesky: Any
dependent: Any
dependent_property: Any
greater_than: Any
greater_than_eq: Any
half_open_interval: Any
independent: Any
integer_interval: Any
is_dependent: Any
less_than: Any
lower_cholesky: Any
lower_triangular: Any
multinomial: Any
nonnegative: Any
nonnegative_integer: Any
one_hot: Any
positive: Any
positive_definite: Any
positive_integer: Any
positive_semidefinite: Any
real_vector: Any
simplex: Any
square: Any
stack: Any
symmetric: Any
unit_interval: Any
