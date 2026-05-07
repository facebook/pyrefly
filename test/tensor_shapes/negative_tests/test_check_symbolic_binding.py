# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test what happens when we use wrong expected type"""

from typing import reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def accepts_symbolic_returns_symbolic[N](x: Tensor[N, 3]) -> Tensor[N, 3]:
    """Identity function with symbolic dimension - preserves shape"""
    return x


def test_symbolic_identity_correct() -> Tensor[2, 3]:
    """This should work - expected type matches"""
    x_concrete: Tensor[2, 3] = torch.randn(2, 3)
    result = accepts_symbolic_returns_symbolic(x_concrete)
    reveal_type(result)
    return result  # Should be OK


def test_symbolic_identity_wrong() -> Tensor[4, 3]:
    """This should ERROR - expected type doesn't match"""
    x_concrete: Tensor[2, 3] = torch.randn(2, 3)
    result = accepts_symbolic_returns_symbolic(x_concrete)
    reveal_type(result)
    return result  # Should ERROR: Tensor[2, 3] not assignable to Tensor[4, 3]


def numel_returns_bad_explicit_symint[N, M](x: Tensor[N, M]) -> Dim[N + M]:
    s = x.numel()
    reveal_type(s)
    return s


def view_returns_bad_explicit_tensor[N, M](x: Tensor[N, M]) -> Tensor[N + M]:
    v = x.view(-1)
    reveal_type(v)
    return v


def numel_returns_bad_implicit_symint[N, M, K](x: Tensor[N, M]) -> Dim[K]:
    s = x.numel()
    reveal_type(s)
    return s


def view_returns_bad_implicit_tensor[N, M, K](x: Tensor[N, M]) -> Tensor[K]:
    v = x.view(-1)
    reveal_type(v)
    return v


def test_numel_returns_bad_implicit_symint() -> Dim[11]:
    n = numel_returns_bad_implicit_symint(torch.randn(3, 4))
    reveal_type(n)
    # Should infer: Literal[12] (3*4=12)
    return n


def test_view_returns_bad_implicit_tensor() -> Tensor[11]:
    t = view_returns_bad_implicit_tensor(torch.randn(3, 4))
    reveal_type(t)
    # Should infer: Literal[12] (3*4=12)
    return t
