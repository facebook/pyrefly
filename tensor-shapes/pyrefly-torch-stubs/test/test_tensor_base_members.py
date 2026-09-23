# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, assert_type

import torch
from shape_extensions import assert_shape
from torch import Tensor
from torch._C import TensorBase


def test_tensor_base_members(tensor: Tensor) -> None:
    base: TensorBase = tensor
    assert_type(base, TensorBase)
    assert_type(tensor.abs_(), Tensor)
    assert_type(tensor.acos_(), Tensor)
    assert_type(tensor.fill_(0), Tensor)
    assert_type(tensor.zero_(), Tensor)
    assert_type(tensor.add_(1.0), Tensor)
    assert_type(tensor.pin_memory(), Tensor)
    assert_type(tensor.byte(), Tensor)
    assert_type(tensor.unique(), Any)
    assert_type(tensor.numpy(), Any)
    assert_type(tensor.data_ptr(), int)
    assert_type(tensor.is_contiguous(), bool)
    assert_type(tensor.grad, Any)
    assert_type(tensor.is_cuda, bool)
    assert_type(float(torch.zeros(())), float)
    assert_type(int(torch.zeros(())), int)
    assert_type(len(tensor), int)


def test_shape_preserving_tensor_base_members() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.fill_(0).shape, (2, 3))
    assert_shape(tensor.clone().shape, (2, 3))
    assert_shape(tensor.detach().shape, (2, 3))
    assert_shape(tensor.transpose(0, 1).contiguous().shape, (3, 2))
