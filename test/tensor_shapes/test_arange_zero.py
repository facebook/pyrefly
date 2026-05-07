# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_arange():
    x = torch.arange(0, 3)
    assert_type(x, Tensor[3])


def test_arange_symbolic[N](t: Dim[N]):
    x = torch.arange(0, t)
    assert_type(x, Tensor[N])


def test_arange_single_arg[N](t: Dim[N]):
    x = torch.arange(t)
    assert_type(x, Tensor[N])
