# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor

if TYPE_CHECKING:
    from shape_extensions import IntVar


def concat_channels[Batch: IntVar, Left: IntVar, Right: IntVar](
    x: Tensor[[Batch, Left]], y: Tensor[[Batch, Right]]
) -> Tensor[[Batch, Left + Right]]:
    return torch.concatenate((x, y), axis=1)


joined = concat_channels(torch.randn((2, 3)), torch.randn((2, 5)))
assert_type(joined, Tensor[[2, 8]])

diagonal = torch.diagonal(torch.ones((6,)), offset=-2)
assert_type(diagonal, Tensor[[8, 8]])

if TYPE_CHECKING:
    bad: Tensor[[2, 9]] = joined  # E: is not assignable
