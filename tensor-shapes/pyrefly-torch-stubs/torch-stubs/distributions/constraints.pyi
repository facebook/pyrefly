# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.constraints."""

class Constraint:
    """Base class for constraints."""

    ...

real: Constraint

def interval(lower_bound: float, upper_bound: float) -> Constraint: ...
