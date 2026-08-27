# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.functional operations with generic TypeVarTuple signatures."""

from typing import assert_type, cast

from shape_extensions import IntTuple, IntVar
from torch import Tensor
from torch.nn import functional as F


def test_activation_functions() -> None:
    """Test activation functions preserve shape via generic signatures."""
    x = cast(Tensor[[2, 3, 4]], ...)

    # Test updated activation functions
    assert_type(F.gelu(x), Tensor[[2, 3, 4]])
    assert_type(F.silu(x), Tensor[[2, 3, 4]])
    assert_type(F.selu(x), Tensor[[2, 3, 4]])
    assert_type(F.elu(x), Tensor[[2, 3, 4]])
    assert_type(F.leaky_relu(x), Tensor[[2, 3, 4]])
    assert_type(F.relu6(x), Tensor[[2, 3, 4]])
    assert_type(F.softplus(x), Tensor[[2, 3, 4]])
    assert_type(F.softsign(x), Tensor[[2, 3, 4]])
    assert_type(F.hardtanh(x), Tensor[[2, 3, 4]])
    assert_type(F.hardsigmoid(x), Tensor[[2, 3, 4]])
    assert_type(F.hardswish(x), Tensor[[2, 3, 4]])
    assert_type(F.sigmoid(x), Tensor[[2, 3, 4]])
    assert_type(F.tanh(x), Tensor[[2, 3, 4]])
    assert_type(F.mish(x), Tensor[[2, 3, 4]])
    assert_type(F.relu(x), Tensor[[2, 3, 4]])


def test_parametric_activations() -> None:
    """Test parametric activation functions."""
    x = cast(Tensor[[2, 3, 4]], ...)
    weight = cast(Tensor, ...)

    assert_type(F.prelu(x, weight), Tensor[[2, 3, 4]])
    assert_type(F.rrelu(x), Tensor[[2, 3, 4]])
    assert_type(F.celu(x), Tensor[[2, 3, 4]])


def test_normalization_functions() -> None:
    """Test normalization functions preserve shape via generic signatures."""
    x = cast(Tensor[[2, 3, 4, 5]], ...)

    # Test updated normalization functions
    assert_type(F.batch_norm(x, None, None), Tensor[[2, 3, 4, 5]])
    assert_type(F.instance_norm(x), Tensor[[2, 3, 4, 5]])
    assert_type(F.layer_norm(x, (4, 5)), Tensor[[2, 3, 4, 5]])
    assert_type(F.group_norm(x, 3), Tensor[[2, 3, 4, 5]])
    assert_type(F.normalize(x), Tensor[[2, 3, 4, 5]])
    assert_type(F.local_response_norm(x, 3), Tensor[[2, 3, 4, 5]])


def test_dropout_functions() -> None:
    """Test dropout functions preserve shape via generic signatures."""
    x = cast(Tensor[[3, 4, 5]], ...)

    # Test updated dropout functions
    assert_type(F.dropout(x), Tensor[[3, 4, 5]])
    assert_type(F.alpha_dropout(x), Tensor[[3, 4, 5]])
    assert_type(F.feature_alpha_dropout(x), Tensor[[3, 4, 5]])


def test_cosine_similarity_reduction() -> None:
    x = cast(Tensor[[2, 3]], ...)
    rank_three = cast(Tensor[[2, 3, 4]], ...)
    y = cast(Tensor[[1, 3]], ...)
    z = cast(Tensor[[2, 1]], ...)
    scalar = cast(Tensor[[]], ...)

    assert_type(F.cosine_similarity(x, x), Tensor[[2]])
    assert_type(F.cosine_similarity(y, z, dim=0), Tensor[[3]])
    assert_type(F.cosine_similarity(y, z, dim=1), Tensor[[2]])
    assert_type(F.cosine_similarity(rank_three, rank_three, dim=-2), Tensor[[2, 4]])
    assert_type(F.cosine_similarity(x, x, dim=-1), Tensor[[2]])
    assert_type(F.cosine_similarity(scalar, scalar, dim=0), Tensor[[]])
    assert_type(F.cosine_similarity(scalar, scalar, dim=-1), Tensor[[]])


def check_cosine_similarity_symbolic[B: IntVar, M: IntVar, N: IntVar](
    x: Tensor[[B, 1, N]], y: Tensor[[1, M, N]]
) -> None:
    assert_type(F.cosine_similarity(x, y, dim=-1), Tensor[[B, M]])
    assert_type(F.cosine_similarity(x, y, dim=1), Tensor[[B, N]])


def check_cosine_similarity_gradual(
    x: Tensor[[int, int, int]],
    y: Tensor[[1, int, 1]],
    bare: Tensor,
    open_rank: Tensor[IntTuple],
    dim: int,
) -> None:
    assert_type(F.cosine_similarity(x, y, dim=-1), Tensor[[int, int]])
    assert_type(F.cosine_similarity(x, y, dim=dim), Tensor[IntTuple])
    assert_type(F.cosine_similarity(bare, x, dim=-1), Tensor[IntTuple])
    assert_type(F.cosine_similarity(open_rank, x, dim=0), Tensor[IntTuple])
