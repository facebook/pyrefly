# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Arbitrary type parameters are not dimensions.

Scalar shape arguments are typed as type parameters bounded by `Int`, so their
runtime arguments are ordinary values rather than `Int[...]` wrappers. That must
not make every type parameter admissible: only an `IntVar` symbol or a variable
bounded by exactly `Int` names a dimension. Anything else fails the bound check,
and no shape ever comes back carrying the caller's unrelated type parameter.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def test_typevar_overloaded_scalar_arguments[T, S: str](
    x: Tensor[[4, 32]], t: T, s: S
) -> None:
    # E: No matching overload found
    torch.repeat_interleave(x, t, dim=1)
    # E: No matching overload found
    torch.repeat_interleave(x, s, dim=1)


def test_typevar_adaptive_pool_scalar_arguments[T, S: str](
    x: Tensor[[2, 64, 56, 56]], t: T, s: S
) -> None:
    # E: No matching overload found
    F.adaptive_avg_pool2d(x, t)
    # E: No matching overload found
    F.adaptive_max_pool2d(x, s)


def test_typevar_adaptive_pool_tuple[T](x: Tensor[[2, 64, 56, 56]], t: T) -> None:
    # Both tuple elements share one type parameter, so shared inference cannot
    # rescue an argument the bound rejects.
    # E: No matching overload found
    F.adaptive_avg_pool2d(x, (t, t))
    # E: No matching overload found
    F.adaptive_avg_pool3d(x, (t, 7, t))
    # E: No matching overload found
    F.adaptive_max_pool1d(x, (t,))
