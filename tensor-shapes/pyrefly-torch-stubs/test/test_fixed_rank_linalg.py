# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_fixed_rank_linalg_shapes() -> None:
    matrix = torch.ones((3, 4))
    other = torch.ones((4, 5))
    assert_shape(torch.mm(matrix, other).shape, (3, 5))
    assert_shape(matrix.mm(other).shape, (3, 5))

    batch = torch.ones((10, 3, 4))
    other_batch = torch.ones((10, 4, 5))
    assert_shape(torch.bmm(batch, other_batch).shape, (10, 3, 5))
    assert_shape(batch.bmm(other_batch).shape, (10, 3, 5))

    vector = torch.ones(4)
    assert_shape(torch.mv(matrix, vector).shape, (3,))
    assert_shape(matrix.mv(vector).shape, (3,))
    assert_shape(torch.outer(torch.ones(3), torch.ones(5)).shape, (3, 5))

    positive_definite = torch.eye(4)
    assert_shape(positive_definite.cholesky().shape, (4, 4))
    assert_shape(positive_definite.inverse().shape, (4, 4))
    assert_shape(positive_definite.matrix_power(3).shape, (4, 4))

    matrix_batch = torch.eye(3).expand((2, 3, 3))
    result = torch.linalg.slogdet(matrix_batch)
    assert_shape(result.sign.shape, (2,))
    assert_shape(result.logabsdet.shape, (2,))


def test_eigendecomposition_shapes() -> None:
    matrix = torch.eye(4)
    values, vectors = torch.linalg.eig(matrix)
    assert_shape(values.shape, (4,))
    assert_shape(vectors.shape, (4, 4))

    batch = matrix.expand(2, 3, 4, 4)
    values, vectors = torch.linalg.eigh(batch)
    assert_shape(values.shape, (2, 3, 4))
    assert_shape(vectors.shape, (2, 3, 4, 4))
    assert_shape(torch.linalg.eigvals(batch).shape, (2, 3, 4))
    assert_shape(torch.linalg.eigvalsh(batch).shape, (2, 3, 4))


def test_square_matrix_operation_shapes() -> None:
    matrix = torch.eye(4)
    batch = matrix.expand(2, 3, 4, 4)

    assert_shape(torch.linalg.cholesky(matrix).shape, (4, 4))
    assert_shape(torch.linalg.inv(batch).shape, (2, 3, 4, 4))
    assert_shape(torch.linalg.matrix_power(batch, 2).shape, (2, 3, 4, 4))
    assert_shape(torch.linalg.matrix_exp(batch).shape, (2, 3, 4, 4))

    assert_shape(torch.linalg.det(batch).shape, (2, 3))
    assert_shape(torch.logdet(batch).shape, (2, 3))
    sign, logabsdet = torch.linalg.slogdet(batch)
    assert_shape(sign.shape, (2, 3))
    assert_shape(logabsdet.shape, (2, 3))
    assert_shape(torch.linalg.matrix_rank(batch).shape, (2, 3))

    assert_shape(torch.trace(matrix).shape, ())
    with assert_raises(RuntimeError):
        # TODO: BUG: Reject batched inputs to `trace` statically.
        torch.trace(batch)


def test_linear_solver_shapes() -> None:
    coefficients = torch.eye(4)
    right_hand_side = torch.ones((4, 2))

    assert_shape(torch.linalg.solve(coefficients, right_hand_side).shape, (4, 2))
    assert_shape(
        torch.linalg.solve_triangular(coefficients, right_hand_side, upper=True).shape,
        (4, 2),
    )
    assert_shape(
        torch.cholesky_solve(right_hand_side, coefficients).shape,
        (4, 2),
    )


def test_mm_rejects_invalid_inputs() -> None:
    assert_shape(torch.mm(torch.ones((2, 3)), torch.ones((3, 4))).shape, (2, 4))

    with assert_raises(RuntimeError):
        # E: Shape dimension mismatch
        torch.mm(torch.ones((2, 3)), torch.ones((4, 5)))

    with assert_raises(RuntimeError):
        torch.ones(3).mm(torch.ones((3, 4)))  # E: Tensor rank mismatch


def test_bmm_rejects_invalid_inputs() -> None:
    assert_shape(
        torch.bmm(torch.ones((2, 3, 4)), torch.ones((2, 4, 5))).shape,
        (2, 3, 5),
    )

    with assert_raises(RuntimeError):
        # E: Shape dimension mismatch
        torch.bmm(torch.ones((2, 3, 4)), torch.ones((3, 4, 5)))

    with assert_raises(RuntimeError):
        # E: Shape dimension mismatch
        torch.bmm(torch.ones((2, 3, 4)), torch.ones((2, 5, 6)))

    with assert_raises(RuntimeError):
        # E: Tensor rank mismatch
        # E: Tensor rank mismatch
        torch.bmm(torch.ones((3, 4)), torch.ones((3, 4)))


def test_mv_and_outer_reject_invalid_inputs() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(torch.mv(matrix, torch.ones(3)).shape, (2,))

    with assert_raises(RuntimeError):
        torch.mv(matrix, torch.ones(4))  # E: Shape dimension mismatch

    with assert_raises(RuntimeError):
        torch.outer(matrix, torch.ones(3))  # E: Tensor rank mismatch

    with assert_raises(RuntimeError):
        torch.outer(torch.ones(3), torch.ones(()))  # E: Tensor rank mismatch


def test_decompositions_reject_low_rank_inputs() -> None:
    scalar = torch.ones(())
    vector = torch.ones(3)
    assert_shape(vector.shape, (3,))

    with assert_raises(RuntimeError):
        # E: eig requires at least 2D input
        torch.linalg.eig(scalar)
    with assert_raises(RuntimeError):
        # E: eig requires at least 2D input
        torch.linalg.eigh(vector)
    with assert_raises(RuntimeError):
        # E: eigvals requires at least 2D input
        torch.linalg.eigvals(vector)
    with assert_raises(RuntimeError):
        # E: eigvals requires at least 2D input
        torch.linalg.eigvalsh(vector)
    with assert_raises(RuntimeError):
        # E: slogdet requires at least 2D input
        torch.linalg.slogdet(scalar)
    with assert_raises(RuntimeError):
        # E: slogdet requires at least 2D input
        torch.linalg.slogdet(vector)


if TYPE_CHECKING:

    def check_advanced_linalg[Batch: IntTuple, M: IntVar, N: IntVar](
        matrix: Tensor[[*Elements[Batch], M, N]],
        right_hand_side: Tensor[[*Elements[Batch], M, 2]],
    ) -> None:
        values, vectors = torch.linalg.eig(matrix)
        assert_type(values, Tensor[[*Elements[Batch], M]])
        assert_type(vectors, Tensor[[*Elements[Batch], M, N]])
        assert_type(torch.linalg.eigvals(matrix), Tensor[[*Elements[Batch], M]])

        sign, logabsdet = torch.linalg.slogdet(matrix)
        assert_type(sign, Tensor[Batch])
        assert_type(logabsdet, Tensor[Batch])
        assert_type(torch.linalg.det(matrix), Tensor[Batch])
        assert_type(torch.linalg.matrix_rank(matrix), Tensor[Batch])

        assert_type(torch.linalg.cholesky(matrix), Tensor[[*Elements[Batch], M, N]])
        assert_type(torch.linalg.inv(matrix), Tensor[[*Elements[Batch], M, N]])
        assert_type(
            torch.linalg.matrix_power(matrix, 2),
            Tensor[[*Elements[Batch], M, N]],
        )
        assert_type(
            torch.linalg.solve(matrix, right_hand_side),
            Tensor[[*Elements[Batch], M, 2]],
        )

    def check_symbolic[B: IntVar, M: IntVar, N: IntVar, K: IntVar](
        matrix: Tensor[[M, N]],
        other: Tensor[[N, K]],
        batch: Tensor[[B, M, N]],
        other_batch: Tensor[[B, N, K]],
        vector: Tensor[[N]],
    ) -> None:
        assert_type(torch.mm(matrix, other), Tensor[[M, K]])
        assert_type(matrix.mm(other), Tensor[[M, K]])
        assert_type(torch.bmm(batch, other_batch), Tensor[[B, M, K]])
        assert_type(batch.bmm(other_batch), Tensor[[B, M, K]])
        assert_type(torch.mv(matrix, vector), Tensor[[M]])
        assert_type(matrix.mv(vector), Tensor[[M]])
        assert_type(torch.outer(vector, vector), Tensor[[N, N]])
        assert_type(batch.slogdet(), torch.return_types.slogdet[[B]])
        assert_type(batch.slogdet().sign, Tensor[[B]])
        assert_type(torch.linalg.slogdet(batch), torch.return_types.linalg_slogdet[[B]])
        assert_type(torch.linalg.slogdet(batch).logabsdet, Tensor[[B]])

        batch.slogdet().indices  # E: no attribute `indices`
