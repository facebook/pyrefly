# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import reveal_type

import torch


def test_mismatched_contracted_dimensions() -> None:
    # TODO: BUG: This should report that contracted dimensions 3 and 4 do not match.
    reveal_type(torch.randn((2, 3)) @ torch.randn((4, 5)))  # E: Tensor[[2, 5]]
