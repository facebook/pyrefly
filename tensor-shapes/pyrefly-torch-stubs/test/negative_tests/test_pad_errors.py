# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import functional as F


def test_invalid_pad_parameters(x: Tensor[[2, 3]]) -> None:
    F.pad(x, (1, True))  # E: No matching overload
    F.pad(x, (1, 2.0))  # E: No matching overload


def test_invalid_pad_arity(x: Tensor[[2, 3]], scalar: Tensor[[]]) -> None:
    F.pad(x, (1, 2, 3))  # E: pad must have an even number of entries
    F.pad(x, (1, 2, 3, 4, 5, 6))  # E: pad has more padding pairs than input dimensions
    F.pad(scalar, (1, 2))  # E: pad does not support scalar input
