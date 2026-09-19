# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from shape_extensions import assert_raises, assert_shape


def test_tile_rejects_short_negative_repeats() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.shape, (2, 3))

    # TODO: BUG: Reject short negative `tile` repeats statically.
    with assert_raises(RuntimeError):
        torch.tile(tensor, (-1,))
