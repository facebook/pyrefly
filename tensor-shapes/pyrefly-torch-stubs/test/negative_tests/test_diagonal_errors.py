# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import reveal_type

import torch


def test_diagonal_is_gradual() -> None:
    tensor = torch.zeros((2, 3, 4))
    # TODO: BUG: Both diagonal APIs should preserve the computed result shape.
    reveal_type(torch.diagonal(tensor))  # E: revealed type: Tensor
    reveal_type(tensor.diagonal(offset=1, dim1=1, dim2=2))  # E: revealed type: Tensor
