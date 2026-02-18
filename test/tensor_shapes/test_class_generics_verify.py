# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Verify: Do class-level generics work properly?
Testing user's claim that class generics DO propagate to methods
"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# Test 1: Class generic used in method (USER'S PATTERN)
class Foo[N]:
    def bar(self, x: Tensor[N], y: Tensor[N]) -> Tensor[N]:
        """N should be visible from class declaration"""
        return x + y


def test_class_generic():
    """Test that class-level N is visible in method"""
    foo = Foo()
    x: Tensor[5] = torch.randn(5)
    y: Tensor[5] = torch.randn(5)
    z = foo.bar(x, y)
    assert_type(z, Tensor[5])


# Test 2: Class generic + method generic (USER'S PATTERN)
class Bar[N]:
    def baz[M](self, x: Tensor[M], y: Tensor[N]) -> Tensor[M, N]:
        """Both M (method-level) and N (class-level) should be visible"""
        # Use einsum for outer product
        result: Tensor[M, N] = torch.einsum("i,j->ij", x, y)
        return result


def test_class_and_method_generic():
    """Test mixing class-level N and method-level M"""
    bar = Bar()
    x: Tensor[3] = torch.randn(3)
    y: Tensor[5] = torch.randn(5)
    z = bar.baz(x, y)
    assert_type(z, Tensor[3, 5])
