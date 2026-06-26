# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

import torch
from torch.nn import Dropout

if TYPE_CHECKING:
    from torch import Tensor

drop = Dropout(0.0)


def test_dropout():
    x = drop(torch.randn(2, 3))
    assert_type(x, Tensor[2, 3])


def test_arange_symbolic[N, M](t: Tensor[N, M]):
    x = drop(t)
    assert_type(x, Tensor[N, M])
