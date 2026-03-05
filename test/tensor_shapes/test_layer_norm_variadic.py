# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test layer_norm with variadic type parameters."""

from typing import assert_type, cast

from torch import Tensor
from torch.nn import functional as F


def test_layer_norm_with_variadic[Ndim, *Bs]() -> None:
    """Test that layer_norm preserves variadic shape with suffix dimension."""
    input = cast(Tensor[*Bs, Ndim], ...)
    weight = cast(Tensor[Ndim], ...)

    # This should preserve the full shape
    result = F.layer_norm(input, weight.shape, weight, None, 1e-5)
    assert_type(result, Tensor[*Bs, Ndim])


def test_layer_norm_concrete() -> None:
    """Test layer_norm with concrete dimensions."""
    input = cast(Tensor[2, 3, 4], ...)
    weight = cast(Tensor[4], ...)

    # This should work fine
    result = F.layer_norm(input, weight.shape, weight, None, 1e-5)
    assert_type(result, Tensor[2, 3, 4])
