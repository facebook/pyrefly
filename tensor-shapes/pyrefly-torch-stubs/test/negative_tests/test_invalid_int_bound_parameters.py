# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Arbitrary type parameters are not dimensions.

Scalar shape arguments are typed as type parameters bounded by `Int`, so their
runtime arguments are ordinary values rather than `Int[...]` carriers. That must
not make every type parameter admissible: only an `IntVar` symbol or a variable
bounded by exactly `Int` names a dimension. Anything else fails the bound check,
and no shape ever comes back carrying the caller's unrelated type parameter.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def test_unconstrained_typevar_scalar_arguments[T](x: Tensor[[4, 32]], t: T) -> None:
    # E: `T` is not assignable to upper bound `Int[int]` of type variable `K`
    values, _ = torch.topk(x, t)
    # E: `T` is not assignable to upper bound `Int[int]` of type variable `Length`
    narrowed = torch.narrow(x, 1, 0, t)
    # E: `T` is not assignable to upper bound `Int[int]` of type variable `NumSamples`
    sampled = torch.multinomial(x, t)
    # The rejected argument never becomes a dimension. Each rule preserves its
    # known rank and axes while the invalid argument's extent is gradual.
    assert_type(narrowed, Tensor[[4, int]])
    assert_type(values, Tensor[[4, int]])
    assert_type(sampled, Tensor[[4, int]])


def test_str_bounded_typevar_scalar_arguments[S: str](x: Tensor[[4, 32]], s: S) -> None:
    # E: `S` is not assignable to upper bound `Int[int]` of type variable `K`
    torch.topk(x, s)
    # E: `S` is not assignable to upper bound `Int[int]` of type variable `Length`
    torch.narrow(x, 1, 0, s)
    # E: `S` is not assignable to upper bound `Int[int]` of type variable `NumSamples`
    torch.multinomial(x, s)


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
