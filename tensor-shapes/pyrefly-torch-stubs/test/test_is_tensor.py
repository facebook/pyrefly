# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import assert_shape


def test_is_tensor() -> None:
    tensor = torch.ones((2, 3))

    assert_type(torch.is_tensor(tensor), bool)
    assert_shape(tensor.shape, (2, 3))
    assert torch.is_tensor(tensor)
    assert not torch.is_tensor([[1, 2], [3, 4]])
