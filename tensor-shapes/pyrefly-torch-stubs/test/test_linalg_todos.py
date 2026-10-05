# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, cast, TYPE_CHECKING

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
    assert_shape(torch.linalg.matrix_norm(matrix).shape, ())
    assert_shape(torch.linalg.matrix_norm(matrix, keepdim=True).shape, (1, 1))
    assert_shape(torch.linalg.pinv(matrix).shape, (4, 3))
    assert_shape(torch.linalg.vander(torch.ones((2, 3))).shape, (2, 3, 3))
    assert_shape(torch.linalg.vander(torch.ones((2, 3)), N=4).shape, (2, 3, 4))
    assert_shape(torch.linalg.cond(torch.eye(3).expand((2, 3, 3))).shape, (2,))
    assert_shape(
        torch.linalg.cross(torch.ones((2, 1, 3)), torch.ones((1, 4, 3))).shape,
        (2, 4, 3),
    )
    assert_shape(
        torch.linalg.vecdot(torch.ones((2, 1, 3)), torch.ones((1, 4, 3))).shape, (2, 4)
    )
    assert_shape(
        torch.linalg.cross(torch.ones((2, 3, 1)), torch.ones((1, 3, 4)), dim=1).shape,
        (2, 3, 4),
    )
    assert_shape(
        torch.linalg.vecdot(torch.ones((2, 3, 1)), torch.ones((1, 3, 4)), dim=1).shape,
        (2, 4),
    )
    ld, pivots, _ = torch.linalg.ldl_factor_ex(torch.eye(3))
    # Factorization returns are not yet typed in these stubs.
    assert_shape(
        torch.linalg.ldl_solve(
            cast("Tensor[[3, 3]]", ld), cast("Tensor[[3]]", pivots), torch.ones((3, 2))
        ).shape,
        (3, 2),
    )
    lu, lu_pivots = torch.linalg.lu_factor(torch.eye(3))
    lu_matrix = cast("Tensor[[3, 3]]", lu)
    lu_pivots = cast("Tensor[[3]]", lu_pivots)
    assert_shape(
        torch.linalg.lu_solve(lu_matrix, lu_pivots, torch.ones((3, 2))).shape,
        (3, 2),
    )
    assert_shape(
        torch.linalg.lu_solve(
            lu_matrix, lu_pivots, torch.ones((2, 3)), left=False
        ).shape,
        (2, 3),
    )
    assert_shape(torch.linalg.svdvals(torch.eye(3)).shape, (3,))


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
        assert_type(torch.linalg.matrix_norm(matrix), Tensor[Batch])
        assert_type(torch.linalg.pinv(matrix), Tensor[[*Batch, N, M]])
        assert_type(torch.linalg.vander(tau), Tensor[[*Batch, K, K]])
        assert_type(torch.linalg.vander(tau, N=2), Tensor[[*Batch, K, 2]])
        assert_type(torch.linalg.cond(matrix), Tensor[Batch])
        assert_type(torch.linalg.cross(tau, tau), Tensor[[*Batch, K]])
        assert_type(torch.linalg.vecdot(tau, tau), Tensor[Batch])
        assert_type(
            torch.linalg.cross(torch.ones((3, 2)), torch.ones((3, 1)), dim=0),
            Tensor[[3, 2]],
        )
        assert_type(
            torch.linalg.vecdot(torch.ones((2, 3)), torch.ones((2, 1)), dim=-2),
            Tensor[[3]],
        )
        torch.linalg.vecdot(  # E: gufunc
            torch.ones((2, 3)), torch.ones((4, 3)), dim=0
        )
        torch.linalg.cross(  # E: Cannot broadcast dimension
            torch.ones((2, 3)), torch.ones((4, 3))
        )
        torch.linalg.cross(  # E: cross vector dimension must have length 3
            torch.ones((2, 4)), torch.ones((1, 4))
        )
        torch.linalg.cross(  # E: cross vector dimension must have length 3
            torch.ones((4, 2)), torch.ones((4, 1)), dim=0
        )
        torch.linalg.cross(  # E: cross vector dimension must have length 3
            torch.ones((3, 2)), torch.ones((4, 2)), dim=0
        )
        torch.linalg.vecdot(  # E: dimension out of range
            torch.ones((2, 3)), torch.ones((2, 3)), dim=2
        )
        assert_type(torch.linalg.svdvals(matrix), Tensor[[*Batch, int]])
        assert_type(torch.linalg.svdvals(torch.eye(3)), Tensor[[3]])

    def check_linalg_solvers[Batch: IntTuple, N: IntVar, K: IntVar](
        factor: Tensor[[*Batch, N, N]],
        pivots: Tensor[[*Batch, N]],
        rhs: Tensor[[*Batch, N, K]],
        rhs_right: Tensor[[*Batch, K, N]],
    ) -> None:
        assert_type(torch.linalg.ldl_solve(factor, pivots, rhs), Tensor[[*Batch, N, K]])
        assert_type(torch.linalg.lu_solve(factor, pivots, rhs), Tensor[[*Batch, N, K]])
        assert_type(
            torch.linalg.lu_solve(factor, pivots, rhs_right, left=False),
            Tensor[[*Batch, K, N]],
        )
        torch.linalg.lu_solve(  # E: gufunc
            torch.ones((3, 3)), torch.ones((3,)), torch.ones((4, 2))
        )
        torch.linalg.lu_solve(  # E: gufunc
            torch.ones((3, 3)), torch.ones((2,)), torch.ones((2, 3)), left=False
        )
