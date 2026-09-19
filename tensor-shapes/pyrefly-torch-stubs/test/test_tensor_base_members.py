# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

from torch import Tensor
from torch._C import TensorBase


def test_tensor_base_members(tensor: Tensor) -> None:
    base: TensorBase = tensor
    assert_type(base, TensorBase)
    assert_type(tensor.abs_(), Tensor)
    assert_type(tensor.acos_(), Tensor)
    assert_type(tensor.is_cuda, bool)
