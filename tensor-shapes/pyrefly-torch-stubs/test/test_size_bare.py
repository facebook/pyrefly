# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check what size() returns for bare Tensor"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_size_on_bare_tensor(x: Tensor):
    """What does size() return?"""
    s = x.size(0)
    assert_type(s, int)
    n: int = 1
    sn = x.size(n)
    assert_type(sn, int)
