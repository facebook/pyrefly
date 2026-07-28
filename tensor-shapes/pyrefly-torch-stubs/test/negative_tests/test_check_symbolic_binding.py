# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test what happens when we use wrong expected type"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


def accepts_symbolic_returns_symbolic[N: IntVar](
    x: Tensor[[N, 3]],
) -> Tensor[[N, 3]]:
    """Identity function with symbolic dimension - preserves shape"""
    return x


def test_symbolic_identity_correct() -> Tensor[[2, 3]]:
    """Expected type matches the preserved concrete shape."""
    x_concrete: Tensor[[2, 3]] = torch.randn(2, 3)
    result = accepts_symbolic_returns_symbolic(x_concrete)
    assert_type(result, Tensor[[2, 3]])
    return result  # Should be OK


def test_symbolic_identity_wrong() -> Tensor[[4, 3]]:
    """Expected type does not match the preserved concrete shape."""
    x_concrete: Tensor[[2, 3]] = torch.randn(2, 3)
    result = accepts_symbolic_returns_symbolic(x_concrete)
    assert_type(result, Tensor[[2, 3]])
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[4, 3]]`
    return result


def numel_returns_bad_explicit_int[N: IntVar, M: IntVar](
    x: Tensor[[N, M]],
) -> Int[N + M]:
    s = x.numel()
    assert_type(s, Int[N * M])
    # E: Returned type `Int[(N * M)]` is not assignable
    #    to declared return type `Int[(N + M)]`
    return s


def view_returns_bad_explicit_tensor[N: IntVar, M: IntVar](
    x: Tensor[[N, M]],
) -> Tensor[[N + M]]:
    v = x.view(-1)
    assert_type(v, Tensor[[N * M]])
    # E: Returned type `Tensor[[(N * M)]]` is not assignable
    #    to declared return type `Tensor[[(N + M)]]`
    return v


def numel_returns_bad_implicit_int[N: IntVar, M: IntVar, K: IntVar](
    x: Tensor[[N, M]],
) -> Int[K]:
    s = x.numel()
    assert_type(s, Int[N * M])
    # E: Returned type `Int[(N * M)]` is not assignable
    #    to declared return type `Int[K]`
    return s


def view_returns_bad_implicit_tensor[N: IntVar, M: IntVar, K: IntVar](
    x: Tensor[[N, M]],
) -> Tensor[[K]]:
    v = x.view(-1)
    assert_type(v, Tensor[[N * M]])
    # E: Returned type `Tensor[[(N * M)]]` is not assignable
    #    to declared return type `Tensor[[K]]`
    return v


def test_numel_returns_bad_implicit_int() -> Int[11]:
    n = numel_returns_bad_implicit_int(torch.randn(3, 4))
    assert_type(n, Int)
    # Should infer: Literal[12] (3*4=12)
    return n


def test_view_returns_bad_implicit_tensor() -> Tensor[[11]]:
    t = view_returns_bad_implicit_tensor(torch.randn(3, 4))
    assert_type(t, Tensor[[Any]])
    # Should infer: Literal[12] (3*4=12)
    return t
