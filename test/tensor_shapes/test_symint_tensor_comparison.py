# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, reveal_type

import torch
from torch import Tensor
from torch_shapes import Dim


# Dim tests
def numel_returns_implicit_symint[N, M](x: Tensor[N, M]):
    s = x.numel()
    reveal_type(s)
    return s


def test_numel_returns_implicit_symint():
    n = numel_returns_implicit_symint(torch.randn(3, 4))
    reveal_type(n)
    # Should infer: Literal[12] (3*4=12)
    assert_type(n, Dim[12])


# Tensor tests
def view_returns_implicit_tensor[N, M](x: Tensor[N, M]):
    v = x.view(-1)
    reveal_type(v)
    return v


def test_view_returns_implicit_tensor():
    t = view_returns_implicit_tensor(torch.randn(3, 4))
    reveal_type(t)
    # Should infer: Literal[12] (3*4=12)
    assert_type(t, Tensor[12])
