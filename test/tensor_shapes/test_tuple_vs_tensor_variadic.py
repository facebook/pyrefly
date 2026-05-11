# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare variadic tuple vs tensor behavior"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# Test with tuple types
def tuple_identity[*Ts](x: tuple[*Ts]) -> tuple[*Ts]:
    return x


class Foo[X]:
    def __init__(self, x: X):
        self.x = x


t = (Foo(0), Foo("h"))
assert_type(t, tuple[Foo[int], Foo[str]])
t_ = tuple_identity(t)
assert_type(t_, tuple[Foo[int], Foo[str]])


# Test with tensor types
def tensor_identity[*Ts](x: Tensor[*Ts]) -> Tensor[*Ts]:
    return x


x: Tensor[10, 20] = torch.randn(10, 20)
assert_type(x, Tensor[10, 20])
y = tensor_identity(x)
assert_type(y, Tensor[10, 20])
