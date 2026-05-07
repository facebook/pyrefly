# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Confirm that #dim is treated identically to dim (strict equality).

The # prefix is stripped during parsing: #batch and batch map to the
same Quantified in the jaxtyping_dims cache. This means:
- #batch requires exact size equality, same as batch
- #batch and batch are interchangeable in annotations
- No broadcast semantics are enforced

Broadcasting is a property of operations (x + y), not annotations.
Size(1) broadcasts to any named dim on element-wise ops, regardless
of whether the annotation uses # or not."""

from typing import assert_type

from jaxtyping import Shaped
from torch import Tensor


def test_broadcast_equals_plain(
    x: Shaped[Tensor, "#batch 3"],
) -> Shaped[Tensor, "batch 3"]:
    """#batch and batch are the same dim — return type with batch accepts #batch input."""
    assert_type(x, Shaped[Tensor, "batch 3"])
    return x


def test_broadcast_strict_equality(
    x: Shaped[Tensor, "#batch 3"],
    y: Shaped[Tensor, "#batch 3"],
) -> Shaped[Tensor, "#batch 3"]:
    """Both params share #batch — currently requires exact equality, not broadcast."""
    return x


def test_named_dim_broadcasts_with_1(
    x: Shaped[Tensor, "batch 3"],
    y: Shaped[Tensor, "1 3"],
) -> None:
    """Size(1) broadcasts to named dim on element-wise ops."""
    assert_type(x + y, Shaped[Tensor, "batch 3"])


def test_hash_dim_broadcasts_with_1(
    x: Shaped[Tensor, "#batch 3"],
    y: Shaped[Tensor, "1 3"],
) -> None:
    """Same behavior with #batch — the # has no effect on broadcasting."""
    assert_type(x + y, Shaped[Tensor, "batch 3"])
