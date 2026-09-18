# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.linalg
from torch import Tensor


def test_eig_rank_errors(scalar: Tensor[[]], vector: Tensor[[3]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: eig requires at least 2D input, got 0D tensor
    torch.linalg.eig(scalar)
    # E: Cannot evaluate type-level shape DSL call: eig requires at least 2D input, got 1D tensor
    torch.linalg.eig(vector)
    # E: Cannot evaluate type-level shape DSL call: eig requires at least 2D input, got 1D tensor
    torch.linalg.eigh(vector)
    # E: Cannot evaluate type-level shape DSL call: eig requires at least 2D input, got 1D tensor
    torch.eig(vector)
    # E: Cannot evaluate type-level shape DSL call: eig requires at least 2D input, got 1D tensor
    torch.eigh(vector)


def test_eigvals_rank_errors(scalar: Tensor[[]], vector: Tensor[[3]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: eigvals requires at least 2D input, got 0D tensor
    torch.linalg.eigvals(scalar)
    # E: Cannot evaluate type-level shape DSL call: eigvals requires at least 2D input, got 1D tensor
    torch.linalg.eigvals(vector)
    # E: Cannot evaluate type-level shape DSL call: eigvals requires at least 2D input, got 1D tensor
    torch.linalg.eigvalsh(vector)


def test_slogdet_rank_errors(scalar: Tensor[[]], vector: Tensor[[3]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: slogdet requires at least 2D input, got 0D tensor
    torch.linalg.slogdet(scalar)
    # E: Cannot evaluate type-level shape DSL call: slogdet requires at least 2D input, got 1D tensor
    torch.linalg.slogdet(vector)
    # E: Cannot evaluate type-level shape DSL call: slogdet requires at least 2D input, got 1D tensor
    torch.slogdet(vector)
    # E: Cannot evaluate type-level shape DSL call: slogdet requires at least 2D input, got 1D tensor
    vector.slogdet()


def test_mv_errors(
    scalar: Tensor[[]],
    vector: Tensor[[3]],
    other_vector: Tensor[[4]],
    matrix: Tensor[[2, 3]],
) -> None:
    # E: Tensor rank mismatch: expected 2 dimensions, got 1 dimensions
    torch.mv(vector, vector)
    # E: Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
    matrix.mv(matrix)
    # E: Shape dimension mismatch: expected Int[3], got Int[4]
    torch.mv(matrix, other_vector)
    # E: Tensor rank mismatch: expected 2 dimensions, got 0 dimensions
    torch.mv(scalar, vector)


def test_outer_rank_errors(
    scalar: Tensor[[]], vector: Tensor[[3]], matrix: Tensor[[2, 3]]
) -> None:
    # E: Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
    torch.outer(matrix, vector)
    # E: Tensor rank mismatch: expected 1 dimensions, got 0 dimensions
    torch.outer(vector, scalar)
