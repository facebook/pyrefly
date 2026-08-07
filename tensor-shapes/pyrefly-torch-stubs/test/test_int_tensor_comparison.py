# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import Int, IntVar
from torch import Tensor


# Dim tests
def numel_returns_implicit_int[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    s = x.numel()
    assert_type(s, Int[N * M])
    return s


def test_numel_returns_implicit_int():
    n = numel_returns_implicit_int(torch.randn(3, 4))
    # Should infer: Literal[12] (3*4=12)
    assert_type(n, Int[12])


# Tensor tests
def view_returns_implicit_tensor[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    v = x.view(-1)
    assert_type(v, Tensor[[N * M]])
    return v


def test_view_returns_implicit_tensor():
    t = view_returns_implicit_tensor(torch.randn(3, 4))
    # Should infer: Literal[12] (3*4=12)
    assert_type(t, Tensor[[12]])
