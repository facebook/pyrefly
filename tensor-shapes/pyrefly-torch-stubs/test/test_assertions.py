# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from shape_extensions import assert_raises, assert_shape


def test_assert_raises_runtime_contract() -> None:
    assert_shape(torch.zeros(1).shape, (1,))

    with assert_raises(ValueError):
        raise ValueError

    with assert_raises(AssertionError):
        with assert_raises(ValueError):
            pass

    with assert_raises(TypeError):
        with assert_raises(ValueError):
            raise TypeError
