# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test concat and flatten actual return types."""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def concat_symbolic[N: IntVar, M: IntVar](
    x: Tensor[[N, 3]], y: Tensor[[M, 3]]
) -> Tensor[[N + M, 3]]:
    """Concat with symbolic dimension addition: N + M"""
    assert_type(x, Tensor[[N, 3]])
    assert_type(y, Tensor[[M, 3]])
    z = torch.cat((x, y), dim=0)
    assert_type(z, Tensor[[N + M, 3]])
    return z


def flatten_symbolic[B: IntVar, N: IntVar, M: IntVar](
    x: Tensor[[B, N, M]],
) -> Tensor[[B * N * M]]:
    """Flatten with symbolic dimension multiplication"""
    assert_type(x, Tensor[[B, N, M]])
    return x.flatten()


def test_concat_what_is_actual_type() -> Tensor[[100, 3]]:
    """What type does concat actually return?"""
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    y: Tensor[[5, 3]] = torch.randn(5, 3)
    z = concat_symbolic(x, y)
    assert_type(z, Tensor[[7, 3]])

    # E: Returned type `Tensor[[7, 3]]` is not assignable
    #    to declared return type `Tensor[[100, 3]]`
    return z


def test_flatten_what_is_actual_type() -> Tensor[[999]]:
    """What type does flatten actually return?"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    y = flatten_symbolic(x)
    assert_type(y, Tensor[[24]])

    # E: Returned type `Tensor[[24]]` is not assignable
    #    to declared return type `Tensor[[999]]`
    return y
