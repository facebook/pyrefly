# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for jaxtyping variadic *name dims — the most common dim feature
in production code (~60+ files use *batch) with zero prior test coverage."""

from typing import assert_type

import torch
from jaxtyping import Float, Shaped
from shape_extensions import static_jaxtyping
from torch import Tensor


@static_jaxtyping("dim *batch")
def test_variadic_passthrough(
    x: Shaped[Tensor, "*batch dim"],
) -> Shaped[Tensor, "*batch dim"]:
    """Variadic *name with a trailing named dim."""
    assert_type(x, Shaped[Tensor, "*batch dim"])
    return x


@static_jaxtyping("*batch")
def test_variadic_trailing_fixed(
    x: Shaped[Tensor, "*batch 3"],
) -> Shaped[Tensor, "*batch 3"]:
    """Variadic *name with a trailing integer dim."""
    assert_type(x, Shaped[Tensor, "*batch 3"])
    return x


@static_jaxtyping("channels *batch")
def test_variadic_both_ends(
    x: Shaped[Tensor, "channels *batch 3"],
) -> Shaped[Tensor, "channels *batch 3"]:
    """Variadic *name with prefix and suffix dims."""
    assert_type(x, Shaped[Tensor, "channels *batch 3"])
    return x


@static_jaxtyping("*batch")
def test_variadic_sin(
    x: Float[Tensor, "*batch 3"],
) -> Float[Tensor, "*batch 3"]:
    """Fixture op (Self return) preserves variadic shape."""
    result = x.sin()
    assert_type(result, Shaped[Tensor, "*batch 3"])
    return result


@static_jaxtyping("m n *batch")
def test_variadic_det(
    x: Shaped[Tensor, "*batch m n"],
) -> None:
    """Fixture op (det) transforms shape: [*batch, m, n] -> [*batch]."""
    result = torch.det(x)
    assert_type(result, Shaped[Tensor, "*batch"])


assert_type(test_variadic_passthrough(torch.randn(2, 3, 4)), Tensor[[2, 3, 4]])
