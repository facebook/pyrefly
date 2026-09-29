# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import assert_shape


def test_float_dtype_alias() -> None:
    assert_type(torch.float, torch.dtype)
    assert_type(torch.pi, float)
    assert torch.float is torch.float32
    assert isinstance(torch.float, torch.dtype)
    assert_shape(torch.ones((2, 3), dtype=torch.float).shape, (2, 3))
