# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.functional operations with generic TypeVarTuple signatures."""

from typing import assert_type, cast

from torch import Tensor
from torch.nn import functional as F


def test_activation_functions() -> None:
    """Test activation functions preserve shape via generic signatures."""
    x = cast(Tensor[2, 3, 4], ...)

    # Test updated activation functions
    assert_type(F.gelu(x), Tensor[2, 3, 4])
    assert_type(F.silu(x), Tensor[2, 3, 4])
    assert_type(F.selu(x), Tensor[2, 3, 4])
    assert_type(F.elu(x), Tensor[2, 3, 4])
    assert_type(F.leaky_relu(x), Tensor[2, 3, 4])
    assert_type(F.relu6(x), Tensor[2, 3, 4])
    assert_type(F.softplus(x), Tensor[2, 3, 4])
    assert_type(F.softsign(x), Tensor[2, 3, 4])
    assert_type(F.hardtanh(x), Tensor[2, 3, 4])
    assert_type(F.hardsigmoid(x), Tensor[2, 3, 4])
    assert_type(F.hardswish(x), Tensor[2, 3, 4])
    assert_type(F.sigmoid(x), Tensor[2, 3, 4])
    assert_type(F.tanh(x), Tensor[2, 3, 4])
    assert_type(F.mish(x), Tensor[2, 3, 4])
    assert_type(F.relu(x), Tensor[2, 3, 4])


def test_parametric_activations() -> None:
    """Test parametric activation functions."""
    x = cast(Tensor[2, 3, 4], ...)
    weight = cast(Tensor, ...)

    assert_type(F.prelu(x, weight), Tensor[2, 3, 4])
    assert_type(F.rrelu(x), Tensor[2, 3, 4])
    assert_type(F.celu(x), Tensor[2, 3, 4])


def test_normalization_functions() -> None:
    """Test normalization functions preserve shape via generic signatures."""
    x = cast(Tensor[2, 3, 4, 5], ...)

    # Test updated normalization functions
    assert_type(F.batch_norm(x, None, None), Tensor[2, 3, 4, 5])
    assert_type(F.instance_norm(x), Tensor[2, 3, 4, 5])
    assert_type(F.layer_norm(x, (4, 5)), Tensor[2, 3, 4, 5])
    assert_type(F.group_norm(x, 3), Tensor[2, 3, 4, 5])
    assert_type(F.normalize(x), Tensor[2, 3, 4, 5])
    assert_type(F.local_response_norm(x, 3), Tensor[2, 3, 4, 5])


def test_dropout_functions() -> None:
    """Test dropout functions preserve shape via generic signatures."""
    x = cast(Tensor[3, 4, 5], ...)

    # Test updated dropout functions
    assert_type(F.dropout(x), Tensor[3, 4, 5])
    assert_type(F.alpha_dropout(x), Tensor[3, 4, 5])
    assert_type(F.feature_alpha_dropout(x), Tensor[3, 4, 5])
