# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that assert_type works with Tensor types for automated type checking"""

from typing import Any, assert_type, reveal_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class LinearLayer[N, M](nn.Module):
    """A simple linear layer with dimension tracking"""

    weight: Tensor[M, N]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        """Forward pass with batch dimension B"""
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        return torch.matmul(x, weight_t)


# Test method-level generics with assert_type
layer = LinearLayer(5, 10)
x: Tensor[16, 5] = torch.randn(16, 5)
y = layer(x)

# This assertion passes - y has the correct inferred type
assert_type(y, Tensor[16, 10])


# Test nested calls
class TwoLayer[N, M, K](nn.Module):
    layer1: LinearLayer[N, M]
    layer2: LinearLayer[M, K]

    def __init__(self, n: Dim[N], m: Dim[M], k: Dim[K]):
        super().__init__()
        self.layer1 = LinearLayer(n, m)
        self.layer2 = LinearLayer(m, k)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        h = self.layer1(x)
        return self.layer2(h)


model = TwoLayer(5, 10, 10)
x2: Tensor[16, 5] = torch.randn(16, 5)
y2 = model(x2)

# This assertion also passes - nested method calls work correctly
assert_type(y2, Tensor[16, 10])

x3 = torch.randn(16, 5)
reveal_type(x3)  # should be Tensor[16, 5]
x4: Tensor = x3
reveal_type(x4)  # should be Tensor
x5: Tensor[16, Any] = x4
reveal_type(x5)  # should be Tensor[16, Any]

"""
x6: Tensor[17, Any] = x5
x7: Tensor[17, Any] = x4
x8: Tensor[17, Any] = x3

x9: Tensor[8 + 8, Any] = x3
reveal_type(x9)

def symbolic_math[N](x: Tensor[N]) -> Tensor[N+1]:
    return x

answer = symbolic_math(torch.randn(4))
reveal_type(answer)
"""
