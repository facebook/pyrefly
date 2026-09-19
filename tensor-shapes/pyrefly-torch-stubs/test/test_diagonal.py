# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from torch import Tensor


def test_diagonal_shape() -> None:
    matrix = torch.zeros((4, 5))
    assert_type(torch.diagonal(matrix), Tensor[[4]])
    assert_type(torch.diagonal(matrix, offset=2), Tensor[[3]])
    assert_type(matrix.diagonal(offset=-2), Tensor[[2]])
    assert_type(torch.diagonal(matrix, offset=6), Tensor[[0]])

    tensor = torch.zeros((2, 3, 4))
    assert_type(torch.diagonal(tensor), Tensor[[4, 2]])
    assert_type(torch.diagonal(tensor, dim1=0, dim2=2), Tensor[[3, 2]])
    assert_type(tensor.diagonal(offset=1, dim1=1, dim2=2), Tensor[[2, 3]])
    assert_type(tensor.diagonal(dim1=-2, dim2=-1), Tensor[[2, 3]])
