# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_zero_step() -> None:
    torch.arange(0, 5, 0)  # E: arange step must be nonzero
    torch.arange(  # E: arange step must be nonzero
        -9223372036854775808, 9223372036854775807, 0
    )


def test_inconsistent_bounds() -> None:
    torch.arange(-1)  # E: arange bounds are inconsistent with step
    torch.arange(5, 0, 1)  # E: arange bounds are inconsistent with step
    torch.arange(0, 5, -1)  # E: arange bounds are inconsistent with step
