# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test what type C[3 * N] produces"""

from typing import assert_type, reveal_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Linear
    from torch_shapes import Dim


def test_subscript[N](n: Dim[N]):
    # What type does Linear[N, 3 * N] produce?
    reveal_type(Linear[N, (3 * N)])
    x = Linear(n, 3 * n)
    reveal_type(x)
    assert_type(x, Linear[N, (3 * N)])
