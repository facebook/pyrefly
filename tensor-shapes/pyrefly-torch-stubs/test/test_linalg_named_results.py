# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, cast, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import return_types, Tensor


def test_cholesky_ex_named_result() -> None:
    result = torch.linalg.cholesky_ex(torch.eye(3).expand((2, 3, 3)))
    factor, info = result
    assert_shape(factor.shape, (2, 3, 3))
    assert_shape(info.shape, (2,))
    assert_shape(result.L.shape, (2, 3, 3))
    assert_shape(result.info.shape, (2,))


def test_inv_ex_named_result() -> None:
    result = torch.linalg.inv_ex(torch.eye(3).expand((2, 3, 3)))
    inverse, info = result
    assert_shape(inverse.shape, (2, 3, 3))
    assert_shape(info.shape, (2,))
    assert_shape(result.inverse.shape, (2, 3, 3))
    assert_shape(result.info.shape, (2,))


def test_solve_ex_named_result() -> None:
    result = torch.linalg.solve_ex(
        torch.eye(3).expand((2, 3, 3)), torch.ones((2, 3, 4))
    )
    solution, info = result
    assert_shape(solution.shape, (2, 3, 4))
    assert_shape(info.shape, (2,))
    assert_shape(result.result.shape, (2, 3, 4))
    assert_shape(result.info.shape, (2,))


def test_ldl_factor_named_result() -> None:
    result = torch.linalg.ldl_factor(torch.eye(3).expand((2, 3, 3)))
    factor, pivots = result
    assert_shape(factor.shape, (2, 3, 3))
    assert_shape(pivots.shape, (2, 3))
    assert_shape(result.LD.shape, (2, 3, 3))
    assert_shape(result.pivots.shape, (2, 3))


def test_ldl_factor_ex_named_result() -> None:
    result = torch.linalg.ldl_factor_ex(torch.eye(3).expand((2, 3, 3)))
    factor, pivots, info = result
    assert_shape(factor.shape, (2, 3, 3))
    assert_shape(pivots.shape, (2, 3))
    assert_shape(info.shape, (2,))
    assert_shape(result.LD.shape, (2, 3, 3))
    assert_shape(result.pivots.shape, (2, 3))
    assert_shape(result.info.shape, (2,))


def test_lu_factor_named_result() -> None:
    result = cast(
        "return_types.linalg_lu_factor[[2, 4, 3], [2, 3]]",
        torch.linalg.lu_factor(torch.eye(4, 3).expand((2, 4, 3))),
    )
    factor, pivots = result
    assert_shape(factor.shape, (2, 4, 3))
    assert_shape(pivots.shape, (2, 3))
    assert_shape(result.LU.shape, (2, 4, 3))
    assert_shape(result.pivots.shape, (2, 3))


def test_lu_factor_ex_named_result() -> None:
    result = cast(
        "return_types.linalg_lu_factor_ex[[2, 4, 3], [2, 3], [2]]",
        torch.linalg.lu_factor_ex(torch.eye(4, 3).expand((2, 4, 3))),
    )
    factor, pivots, info = result
    assert_shape(factor.shape, (2, 4, 3))
    assert_shape(pivots.shape, (2, 3))
    assert_shape(info.shape, (2,))
    assert_shape(result.LU.shape, (2, 4, 3))
    assert_shape(result.pivots.shape, (2, 3))
    assert_shape(result.info.shape, (2,))


def test_lu_named_result() -> None:
    result = cast(
        "return_types.linalg_lu[[2, 4, 4], [2, 4, 3], [2, 3, 3]]",
        torch.linalg.lu(torch.eye(4, 3).expand((2, 4, 3))),
    )
    permutation, lower, upper = result
    assert_shape(permutation.shape, (2, 4, 4))
    assert_shape(lower.shape, (2, 4, 3))
    assert_shape(upper.shape, (2, 3, 3))
    assert_shape(result.P.shape, (2, 4, 4))
    assert_shape(result.L.shape, (2, 4, 3))
    assert_shape(result.U.shape, (2, 3, 3))


def test_qr_named_result() -> None:
    result = cast(
        "return_types.linalg_qr[[2, 4, 3], [2, 3, 3]]",
        torch.linalg.qr(torch.eye(4, 3).expand((2, 4, 3)), mode="reduced"),
    )
    orthogonal, triangular = result
    assert_shape(orthogonal.shape, (2, 4, 3))
    assert_shape(triangular.shape, (2, 3, 3))
    assert_shape(result.Q.shape, (2, 4, 3))
    assert_shape(result.R.shape, (2, 3, 3))


def test_svd_named_result() -> None:
    result = cast(
        "return_types.linalg_svd[[2, 4, 3], [2, 3], [2, 3, 3]]",
        torch.linalg.svd(torch.eye(4, 3).expand((2, 4, 3)), full_matrices=False),
    )
    left, singular, right = result
    assert_shape(left.shape, (2, 4, 3))
    assert_shape(singular.shape, (2, 3))
    assert_shape(right.shape, (2, 3, 3))
    assert_shape(result.U.shape, (2, 4, 3))
    assert_shape(result.S.shape, (2, 3))
    assert_shape(result.Vh.shape, (2, 3, 3))


def test_lstsq_named_result() -> None:
    a = torch.eye(4, 3).expand((2, 4, 3))
    b = torch.ones((2, 4, 2))
    result = cast(
        "return_types.linalg_lstsq[[2, 3, 2], [2, 2], [2], [2, 3]]",
        torch.linalg.lstsq(a, b, driver="gelsd"),
    )
    solution, residuals, rank, singular_values = result
    assert_shape(solution.shape, (2, 3, 2))
    assert_shape(residuals.shape, (2, 2))
    assert_shape(rank.shape, (2,))
    assert_shape(singular_values.shape, (2, 3))
    assert_shape(result.solution.shape, (2, 3, 2))
    assert_shape(result.residuals.shape, (2, 2))
    assert_shape(result.rank.shape, (2,))
    assert_shape(result.singular_values.shape, (2, 3))


if TYPE_CHECKING:

    def check_ex_named_results[Matrix: IntTuple, Batch: IntTuple, RHS: IntTuple](
        chol: return_types.linalg_cholesky_ex[Matrix, Batch],
        inverse: return_types.linalg_inv_ex[Matrix, Batch],
        solution: return_types.linalg_solve_ex[RHS, Batch],
    ) -> None:
        assert_type(chol[0], Tensor[Matrix])
        assert_type(chol[1], Tensor[Batch])
        assert_type(chol.L, Tensor[Matrix])
        assert_type(chol.info, Tensor[Batch])
        assert_type(inverse[0], Tensor[Matrix])
        assert_type(inverse[1], Tensor[Batch])
        assert_type(inverse.inverse, Tensor[Matrix])
        assert_type(inverse.info, Tensor[Batch])
        assert_type(solution[0], Tensor[RHS])
        assert_type(solution[1], Tensor[Batch])
        assert_type(solution.result, Tensor[RHS])
        assert_type(solution.info, Tensor[Batch])

    def check_factor_named_results[
        Matrix: IntTuple,
        Pivots: IntTuple,
        Batch: IntTuple,
    ](
        ldl: return_types.linalg_ldl_factor[Matrix, Pivots],
        ldl_ex: return_types.linalg_ldl_factor_ex[Matrix, Pivots, Batch],
        lu: return_types.linalg_lu_factor[Matrix, Pivots],
        lu_ex: return_types.linalg_lu_factor_ex[Matrix, Pivots, Batch],
    ) -> None:
        assert_type(ldl[0], Tensor[Matrix])
        assert_type(ldl[1], Tensor[Pivots])
        assert_type(ldl.LD, Tensor[Matrix])
        assert_type(ldl.pivots, Tensor[Pivots])
        assert_type(ldl_ex[0], Tensor[Matrix])
        assert_type(ldl_ex[1], Tensor[Pivots])
        assert_type(ldl_ex[2], Tensor[Batch])
        assert_type(ldl_ex.LD, Tensor[Matrix])
        assert_type(ldl_ex.pivots, Tensor[Pivots])
        assert_type(ldl_ex.info, Tensor[Batch])
        assert_type(lu[0], Tensor[Matrix])
        assert_type(lu[1], Tensor[Pivots])
        assert_type(lu.LU, Tensor[Matrix])
        assert_type(lu.pivots, Tensor[Pivots])
        assert_type(lu_ex[0], Tensor[Matrix])
        assert_type(lu_ex[1], Tensor[Pivots])
        assert_type(lu_ex[2], Tensor[Batch])
        assert_type(lu_ex.LU, Tensor[Matrix])
        assert_type(lu_ex.pivots, Tensor[Pivots])
        assert_type(lu_ex.info, Tensor[Batch])

    def check_decomposition_named_results[
        P: IntTuple,
        L: IntTuple,
        U: IntTuple,
        Q: IntTuple,
        R: IntTuple,
        S: IntTuple,
        Vh: IntTuple,
    ](
        lu: return_types.linalg_lu[P, L, U],
        qr: return_types.linalg_qr[Q, R],
        svd: return_types.linalg_svd[U, S, Vh],
    ) -> None:
        assert_type(lu[0], Tensor[P])
        assert_type(lu[1], Tensor[L])
        assert_type(lu[2], Tensor[U])
        assert_type(lu.P, Tensor[P])
        assert_type(lu.L, Tensor[L])
        assert_type(lu.U, Tensor[U])
        assert_type(qr[0], Tensor[Q])
        assert_type(qr[1], Tensor[R])
        assert_type(qr.Q, Tensor[Q])
        assert_type(qr.R, Tensor[R])
        assert_type(svd[0], Tensor[U])
        assert_type(svd[1], Tensor[S])
        assert_type(svd[2], Tensor[Vh])
        assert_type(svd.U, Tensor[U])
        assert_type(svd.S, Tensor[S])
        assert_type(svd.Vh, Tensor[Vh])

    def check_lstsq_named_result[
        Solution: IntTuple,
        Residuals: IntTuple,
        Rank: IntTuple,
        Singular: IntTuple,
    ](result: return_types.linalg_lstsq[Solution, Residuals, Rank, Singular]) -> None:
        assert_type(result[0], Tensor[Solution])
        assert_type(result[1], Tensor[Residuals])
        assert_type(result[2], Tensor[Rank])
        assert_type(result[3], Tensor[Singular])
        assert_type(result.solution, Tensor[Solution])
        assert_type(result.residuals, Tensor[Residuals])
        assert_type(result.rank, Tensor[Rank])
        assert_type(result.singular_values, Tensor[Singular])
