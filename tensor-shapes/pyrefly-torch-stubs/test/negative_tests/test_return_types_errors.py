# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that `torch.return_types` tuples expose only their documented fields."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_wrong_field_names(x: Tensor[[4, 3]]) -> None:
    x.sort(dim=0).value  # E: no attribute `value`
    x.aminmax().values  # E: no attribute `values`
    x.slogdet().indices  # E: no attribute `indices`
