# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for expanded jaxtyping dimension syntax.

Covers: anonymous dims (_), ellipsis (...), broadcast (#), combined (*#),
arithmetic (dim+1, n-1), parenthesized arithmetic, scalar, leading space.
"""

from jaxtyping import Shaped
from torch import Tensor


# --- Anonymous dim: _ ---


def test_anonymous_dim(
    x: Shaped[Tensor, "_ _"],
) -> Shaped[Tensor, "_ _"]:
    """_ means any single dimension, not bound to a name."""
    return x


def test_anonymous_dims_independent(
    x: Shaped[Tensor, "_ _"],
    y: Shaped[Tensor, "_ _"],
) -> None:
    """Two _ dims are independent (not enforced to be equal)."""
    pass


def test_anonymous_mixed(
    x: Shaped[Tensor, "batch _ 3"],
) -> Shaped[Tensor, "batch _ 3"]:
    """_ can be mixed with named and integer dims."""
    return x


# --- Ellipsis: ... ---


def test_ellipsis_only(
    x: Shaped[Tensor, "..."],
) -> Shaped[Tensor, "..."]:
    """... alone matches any shape (equivalent to shapeless)."""
    return x


def test_ellipsis_with_suffix(
    x: Shaped[Tensor, "... 3"],
) -> Shaped[Tensor, "... 3"]:
    """... with a suffix: any number of leading dims, last dim is 3."""
    return x


def test_ellipsis_with_prefix(
    x: Shaped[Tensor, "batch ..."],
) -> Shaped[Tensor, "batch ..."]:
    """... with a prefix: first dim is batch, rest is anything."""
    return x


def test_ellipsis_with_both(
    x: Shaped[Tensor, "batch ... channels"],
) -> Shaped[Tensor, "batch ... channels"]:
    """... with prefix and suffix dims."""
    return x


# --- Broadcast: #name ---


def test_broadcast_multiple(
    x: Shaped[Tensor, "#batch #channels 3"],
) -> Shaped[Tensor, "#batch #channels 3"]:
    """Multiple broadcast dims."""
    return x


# --- Combined variadic + broadcast: *#name ---


def test_variadic_broadcast(
    x: Shaped[Tensor, "*#batch 3"],
) -> Shaped[Tensor, "*#batch 3"]:
    """*#name: variadic + broadcast combined."""
    return x


def test_variadic_broadcast_with_suffix(
    x: Shaped[Tensor, "channels *#batch 3"],
) -> Shaped[Tensor, "channels *#batch 3"]:
    """*#name with prefix and suffix."""
    return x


# --- Arithmetic: dim+1, n-1, 1+T ---


def test_arithmetic_add(
    x: Shaped[Tensor, "n n+1"],
) -> Shaped[Tensor, "n n+1"]:
    """Addition in dimension expression."""
    return x


def test_arithmetic_sub(
    x: Shaped[Tensor, "n n-1"],
) -> Shaped[Tensor, "n n-1"]:
    """Subtraction in dimension expression."""
    return x


def test_arithmetic_literal_first(
    x: Shaped[Tensor, "n 1+n"],
) -> Shaped[Tensor, "n 1+n"]:
    """Literal on the left side of arithmetic."""
    return x


def test_arithmetic_two_names(
    x: Shaped[Tensor, "a b a+b"],
) -> Shaped[Tensor, "a b a+b"]:
    """Arithmetic with two named dims."""
    return x


# --- Parenthesized arithmetic ---


def test_paren_arithmetic(
    x: Shaped[Tensor, "n (n+1)"],  # noqa: F821
) -> Shaped[Tensor, "n (n+1)"]:  # noqa: F821
    """Parenthesized arithmetic expression."""
    return x


# --- Scalar: "" ---


def test_scalar_identity(x: Shaped[Tensor, ""]) -> Shaped[Tensor, ""]:
    """Empty shape string means scalar tensor (rank 0)."""
    return x


# --- Leading space ---


def test_leading_space(
    x: Shaped[Tensor, " batch features"],
) -> Shaped[Tensor, "batch features"]:
    """Leading space is trimmed (jaxtyping convention). Same type as no leading space."""
    return x


def test_trailing_space(
    x: Shaped[Tensor, "batch features "],
) -> Shaped[Tensor, "batch features"]:
    """Trailing space is trimmed. Same type as no trailing space."""
    return x
