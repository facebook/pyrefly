# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from shape_extensions import assert_raises, assert_shape


def test_zeros() -> None:
    assert_shape(torch.zeros((2, 3)).shape, (2, 3))
    with assert_raises(TypeError):
        torch.zeros("invalid")  # E: No matching overload
