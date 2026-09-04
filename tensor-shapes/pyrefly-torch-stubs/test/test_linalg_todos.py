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
    assert_shape(
        torch.linalg.ldl_solve(ld, pivots, torch.ones((3, 2))).shape,
        (3, 2),
    )
    lu, lu_pivots = torch.linalg.lu_factor(torch.eye(3))
    assert_shape(
        torch.linalg.lu_solve(lu, lu_pivots, torch.ones((3, 2))).shape,
        (3, 2),
    )
    assert_shape(
        torch.linalg.lu_solve(lu, lu_pivots, torch.ones((2, 3)), left=False).shape,
        (2, 3),
    )
    assert_shape(torch.linalg.svdvals(torch.eye(3)).shape, (3,))
    assert_shape(
        torch.linalg.multi_dot(
            (torch.ones((2, 3)), torch.ones((3, 4)), torch.ones((4, 5)))
        ).shape,
        (2, 5),
    )
    inverse_input = torch.eye(6).reshape((2, 3, 3, 2))
    assert_shape(torch.linalg.tensorinv(inverse_input).shape, (3, 2, 2, 3))
    identity = torch.eye(3).reshape((3, 3))
    assert_shape(torch.linalg.tensorinv(identity, ind=1).shape, (3, 3))
    assert_shape(
        torch.linalg.tensorsolve(inverse_input, torch.ones((2, 3))).shape, (3, 2)
    )
    assert_shape(
        torch.linalg.tensorsolve(inverse_input, torch.ones((3, 2)), dims=(0, 1)).shape,
        (2, 3),
    )
    batch_matrix = torch.eye(3).expand((2, 3, 3))
    chol = torch.linalg.cholesky_ex(batch_matrix)
    assert_shape(chol.L.shape, (2, 3, 3))
    assert_shape(chol.info.shape, (2,))
    inverse = torch.linalg.inv_ex(batch_matrix)
    assert_shape(inverse.inverse.shape, (2, 3, 3))
    assert_shape(inverse.info.shape, (2,))
    solution = torch.linalg.solve_ex(batch_matrix, torch.ones((2, 3, 4)))
    assert_shape(solution.result.shape, (2, 3, 4))
    assert_shape(solution.info.shape, (2,))
    ldl = torch.linalg.ldl_factor(batch_matrix)
    assert_shape(ldl.LD.shape, (2, 3, 3))
    assert_shape(ldl.pivots.shape, (2, 3))
    ldl_ex = torch.linalg.ldl_factor_ex(batch_matrix)
    assert_shape(ldl_ex.info.shape, (2,))
    rectangular = torch.eye(4, 3).expand((2, 4, 3))
    lu = torch.linalg.lu_factor(rectangular)
    assert_shape(lu.LU.shape, (2, 4, 3))
    assert lu.pivots.shape == (2, 3)
    lu_ex = torch.linalg.lu_factor_ex(rectangular)
    assert_shape(lu_ex.info.shape, (2,))
    full_lu = torch.linalg.lu(batch_matrix)
    assert_shape(full_lu.P.shape, (2, 3, 3))
    assert_shape(full_lu.L.shape, (2, 3, 3))
    assert_shape(full_lu.U.shape, (2, 3, 3))
    qr = torch.linalg.qr(batch_matrix)
    assert_shape(qr.Q.shape, (2, 3, 3))
    assert_shape(qr.R.shape, (2, 3, 3))
    assert_shape(torch.linalg.qr(rectangular, mode="complete").Q.shape, (2, 4, 4))
    assert_shape(torch.linalg.qr(rectangular, mode="r").Q.shape, (0,))
    decomposition = torch.linalg.svd(batch_matrix)
    assert_shape(decomposition.U.shape, (2, 3, 3))
    assert_shape(decomposition.S.shape, (2, 3))
    assert_shape(decomposition.Vh.shape, (2, 3, 3))
    assert_shape(torch.linalg.svd(rectangular).U.shape, (2, 4, 4))
    assert_shape(torch.linalg.svd(rectangular).Vh.shape, (2, 3, 3))
    matrix_solution = torch.linalg.lstsq(batch_matrix, torch.ones((2, 3, 4)))
    assert_shape(matrix_solution.solution.shape, (2, 3, 4))
    assert matrix_solution.residuals.shape == (0,)
    assert matrix_solution.rank.shape == (2,)
    assert matrix_solution.singular_values.shape == (0,)
    vector_solution = torch.linalg.lstsq(batch_matrix, torch.ones((2, 3)))
    assert_shape(vector_solution.solution.shape, (2, 3))


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
        assert_type(torch.linalg.common_notes, dict[str, str])
        assert_type(torch.linalg.lu_factor(matrix).pivots, Tensor[[*Batch, int]])

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
        assert_type(
            torch.linalg.cholesky_ex(factor),
            torch.return_types.linalg_cholesky_ex[[*Batch, N, N], Batch],
        )
        assert_type(
            torch.linalg.inv_ex(factor),
            torch.return_types.linalg_inv_ex[[*Batch, N, N], Batch],
        )
        assert_type(
            torch.linalg.solve_ex(factor, rhs),
            torch.return_types.linalg_solve_ex[[*Batch, N, K], Batch],
        )
        assert_type(
            torch.linalg.ldl_factor(factor),
            torch.return_types.linalg_ldl_factor[[*Batch, N, N], [*Batch, N]],
        )
        assert_type(
            torch.linalg.ldl_factor_ex(factor),
            torch.return_types.linalg_ldl_factor_ex[[*Batch, N, N], [*Batch, N], Batch],
        )
        assert_type(
            torch.linalg.lu_factor(factor),
            torch.return_types.linalg_lu_factor[[*Batch, N, N], [*Batch, N]],
        )
        assert_type(
            torch.linalg.lu_factor_ex(factor),
            torch.return_types.linalg_lu_factor_ex[[*Batch, N, N], [*Batch, N], Batch],
        )
        assert_type(
            torch.linalg.lu(factor),
            torch.return_types.linalg_lu[
                [*Batch, N, N], [*Batch, N, N], [*Batch, N, N]
            ],
        )
        assert_type(
            torch.linalg.qr(factor),
            torch.return_types.linalg_qr[[*Batch, N, N], [*Batch, N, N]],
        )
        assert_type(
            torch.linalg.svd(factor, full_matrices=False),
            torch.return_types.linalg_svd[[*Batch, N, N], [*Batch, N], [*Batch, N, N]],
        )
        assert_type(
            torch.linalg.lstsq(factor, rhs),
            torch.return_types.linalg_lstsq[
                [*Batch, N, K], IntTuple, IntTuple, IntTuple
            ],
        )
        assert_type(
            torch.linalg.lstsq(factor, pivots),
            torch.return_types.linalg_lstsq[[*Batch, N], IntTuple, IntTuple, IntTuple],
        )

    def check_linalg_tensor_equations[
        Rows: IntVar,
        Cols: IntVar,
        InnerRows: IntVar,
        InnerCols: IntVar,
    ](
        operator: Tensor[[Rows, Cols, InnerRows, InnerCols]],
        right_hand_side: Tensor[[Rows, Cols]],
    ) -> None:
        assert_type(
            torch.linalg.tensorinv(operator),
            Tensor[[InnerRows, InnerCols, Rows, Cols]],
        )
        torch.linalg.tensorinv(  # E: tensorinv input products must match
            torch.ones((2, 3, 4, 5))
        )
        torch.linalg.tensorinv(  # E: tensorinv ind must be positive
            torch.ones((2, 3)), ind=0
        )
        assert_type(
            torch.linalg.tensorsolve(operator, right_hand_side),
            Tensor[[InnerRows, InnerCols]],
        )
        assert_type(
            torch.linalg.tensorsolve(
                torch.ones((2, 3, 3, 2)), torch.ones((3, 2)), dims=(0, 1)
            ),
            Tensor[[2, 3]],
        )
        torch.linalg.tensorsolve(  # E: tensorsolve operator products must match
            torch.ones((2, 3, 4, 5)), torch.ones((2, 3))
        )
        torch.linalg.tensorsolve(  # E: tensorsolve right-hand side size must match
            torch.ones((2, 3, 3, 2)), torch.ones((5, 1))
        )
        torch.linalg.tensorsolve(  # E: tensorsolve dimension out of range
            torch.ones((2, 3, 3, 2)), torch.ones((2, 3)), dims=(4,)
        )
