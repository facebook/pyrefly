# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for jaxtyping annotation parsing: Float[Tensor, "batch channels"]"""

from typing import assert_type

import torch
from jaxtyping import Float, Shaped
from shape_extensions import static_jaxtyping
from torch import Tensor


@static_jaxtyping("")
def test_concrete_dims():
    """Fixed integer dimensions in jaxtyping annotations."""
    x = torch.randn(3, 4)
    assert_type(x, Shaped[Tensor, "3 4"])


@static_jaxtyping("")
def test_concrete_dims_parameter(x: Float[Tensor, "3 4"]) -> None:
    """Jaxtyping wrappers also parse when used as assert_type expected types."""
    assert_type(x, Tensor[[3, 4]])


@static_jaxtyping("h w *batch")
def test_float_variadic_parameter(x: Float[Tensor, "*batch h w"]) -> None:
    """Float also supports nontrivial jaxtyping shape syntax."""
    assert_type(x, Shaped[Tensor, "*batch h w"])


@static_jaxtyping("dim")
def test_float_arithmetic_parameter(x: Float[Tensor, "dim dim+1"]) -> None:
    """Float also supports jaxtyping dimension arithmetic."""
    assert_type(x, Shaped[Tensor, "dim dim+1"])


@static_jaxtyping("batch channels")
def test_named_dims(
    x: Shaped[Tensor, "batch channels"],
) -> Shaped[Tensor, "batch channels"]:
    """Named dimensions are consistent across parameter and return type."""
    assert_type(x, Shaped[Tensor, "batch channels"])
    return x


@static_jaxtyping("batch")
def test_mixed_dims(x: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 3"]:
    """Mix of named and integer dimensions."""
    assert_type(x, Shaped[Tensor, "batch 3"])
    return x


@static_jaxtyping("batch m p n")
def test_matmul_shapes(
    a: Shaped[Tensor, "batch m n"],
    b: Shaped[Tensor, "batch n p"],
) -> Shaped[Tensor, "batch m p"]:
    """Matrix multiply with named batch, m, n, p dimensions."""
    result = torch.matmul(a, b)
    assert_type(result, Shaped[Tensor, "batch m p"])
    return result


@static_jaxtyping("features")
def test_single_dim(x: Shaped[Tensor, "features"]) -> Shaped[Tensor, "features"]:  # noqa: F821
    """Single named dimension."""
    assert_type(x, Shaped[Tensor, "features"])
    return x


@static_jaxtyping("")
def test_scalar(x: Shaped[Tensor, ""]) -> Shaped[Tensor, ""]:
    """Empty shape string means scalar tensor (rank 0)."""
    assert_type(x, Shaped[Tensor, ""])
    return x


# Pin jaxtyping annotations against the native spelling so both sides cannot
# degrade together without failing the corpus.
assert_type(test_named_dims(torch.randn(2, 3)), Tensor[[2, 3]])
assert_type(test_mixed_dims(torch.randn(2, 3)), Tensor[[2, 3]])
assert_type(test_scalar(torch.randn(())), Tensor[[]])
