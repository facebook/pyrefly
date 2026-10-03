# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple, IntVar
from torch import Tensor


def test_linalg_tensor_operations() -> None:
    matrix = torch.ones((3, 4))
    assert_shape(torch.linalg.matmul(matrix, torch.ones((4, 2))).shape, (3, 2))
    assert_shape(torch.linalg.diagonal(matrix).shape, (3,))
    assert_shape(torch.linalg.diagonal(matrix, offset=1).shape, (3,))

    assert_shape(
        torch.linalg.householder_product(torch.eye(3), torch.ones(3)).shape, (3, 3)
    )


if TYPE_CHECKING:

    def check_linalg_tensor_operations[
        Batch: IntTuple,
        M: IntVar,
        N: IntVar,
        K: IntVar,
    ](
        matrix: Tensor[[*Batch, M, N]],
        tau: Tensor[[*Batch, K]],
        other: Tensor[[*Batch, N, K]],
    ) -> None:
        assert_type(torch.linalg.matmul(matrix, other), Tensor[[*Batch, M, K]])
        assert_type(
            torch.linalg.householder_product(matrix, tau), Tensor[[*Batch, M, N]]
        )
