# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_mismatched_contracted_dimensions() -> None:
    # E: core dimension 'n' has conflicting extents 3 and 4
    torch.randn((2, 3)) @ torch.randn((4, 5))


def test_mismatched_batch_dimensions() -> None:
    # E: Cannot broadcast dimension Int[2] with dimension Int[5] at position 0
    torch.matmul(torch.randn((2, 3, 4)), torch.randn((5, 4, 6)))
