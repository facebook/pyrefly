# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Each jaxtyping form means the same as its native spelling.

Every case pairs a sugared definition with the native one it desugars to and
applies both to the same argument, so the two must agree. Stating it as an
equivalence keeps the test about the sugar rather than about whichever type
happens to be printed, and a divergence names the form that broke.

The shape-string grammar itself is covered by unit tests in
`pyrefly/lib/alt/jaxtyping.rs`, which do not need a checker to run.
"""

from typing import assert_type

from jaxtyping import Float
from shape_extensions import Elements, IntTuple, IntVar, static_jaxtyping
from torch import Tensor

# --- Named dimensions ---


@static_jaxtyping("batch channels")
def sugar_named(
    x: Float[Tensor, "batch channels"],
) -> Float[Tensor, "channels batch"]: ...


def native_named[B: IntVar, C: IntVar](x: Tensor[[B, C]]) -> Tensor[[C, B]]: ...


def check_named(x: Tensor[[2, 3]]) -> None:
    assert_type(sugar_named(x), Tensor[[3, 2]])
    assert_type(native_named(x), Tensor[[3, 2]])


# --- Integer literals ---


@static_jaxtyping("")
def sugar_literal(x: Float[Tensor, "2 3"]) -> Float[Tensor, "3 2"]: ...


def native_literal(x: Tensor[[2, 3]]) -> Tensor[[3, 2]]: ...


def check_literal(x: Tensor[[2, 3]]) -> None:
    assert_type(sugar_literal(x), Tensor[[3, 2]])
    assert_type(native_literal(x), Tensor[[3, 2]])


# --- Arithmetic, including the parenthesized spelling ---


@static_jaxtyping("n")
def sugar_arithmetic(x: Float[Tensor, "n"]) -> Float[Tensor, "n+1"]: ...  # noqa: F821


@static_jaxtyping("n")
def sugar_parenthesized(x: Float[Tensor, "n"]) -> Float[Tensor, "(n+1)"]: ...  # noqa: F821


@static_jaxtyping("n")
def sugar_subtraction(x: Float[Tensor, "n"]) -> Float[Tensor, "n-1"]: ...  # noqa: F821


def native_increment[N: IntVar](x: Tensor[[N]]) -> Tensor[[N + 1]]: ...


def native_decrement[N: IntVar](x: Tensor[[N]]) -> Tensor[[N - 1]]: ...


def check_arithmetic(x: Tensor[[5]]) -> None:
    assert_type(sugar_arithmetic(x), Tensor[[6]])
    assert_type(sugar_parenthesized(x), Tensor[[6]])
    assert_type(native_increment(x), Tensor[[6]])
    assert_type(sugar_subtraction(x), Tensor[[4]])
    assert_type(native_decrement(x), Tensor[[4]])


# --- Variadic shapes ---


@static_jaxtyping("*batch c")
def sugar_variadic(x: Float[Tensor, "*batch c"]) -> Float[Tensor, "*batch"]: ...  # noqa: F821


def native_variadic[B: IntTuple, C: IntVar](
    x: Tensor[[*Elements[B], C]],
) -> Tensor[B]: ...


def check_variadic(x: Tensor[[2, 3, 4]]) -> None:
    assert_type(sugar_variadic(x), Tensor[[2, 3]])
    assert_type(native_variadic(x), Tensor[[2, 3]])


# --- Broadcast `#` is accepted and carries no meaning of its own ---


@static_jaxtyping("batch channels")
def sugar_broadcast(
    x: Float[Tensor, "#batch #channels"],
) -> Float[Tensor, "channels batch"]: ...


def check_broadcast(x: Tensor[[2, 3]]) -> None:
    assert_type(sugar_broadcast(x), Tensor[[3, 2]])
    assert_type(native_named(x), Tensor[[3, 2]])


@static_jaxtyping("batch")
def sugar_operator_broadcast(
    x: Float[Tensor, "batch 3"],
    y: Float[Tensor, "1 3"],
) -> Float[Tensor, "batch 3"]:
    return x + y


@static_jaxtyping("batch")
def sugar_hash_operator_broadcast(
    x: Float[Tensor, "#batch 3"],
    y: Float[Tensor, "1 3"],
) -> Float[Tensor, "batch 3"]:
    return x + y


def check_operator_broadcast(x: Tensor[[2, 3]], y: Tensor[[1, 3]]) -> None:
    assert_type(sugar_operator_broadcast(x, y), Tensor[[2, 3]])
    assert_type(sugar_hash_operator_broadcast(x, y), Tensor[[2, 3]])


@static_jaxtyping("*batch c")
def sugar_broadcast_variadic(
    x: Float[Tensor, "*#batch c"],
) -> Float[Tensor, "*batch"]: ...  # noqa: F821


def check_broadcast_variadic(x: Tensor[[2, 3, 4]]) -> None:
    assert_type(sugar_broadcast_variadic(x), Tensor[[2, 3]])
    assert_type(native_variadic(x), Tensor[[2, 3]])


# --- Anonymous dimensions and the ellipsis ---


@static_jaxtyping("")
def sugar_anonymous(x: Float[Tensor, "_ 3"]) -> Float[Tensor, "3"]: ...


def native_anonymous(x: Tensor[[int, 3]]) -> Tensor[[3]]: ...


def check_anonymous(x: Tensor[[2, 3]]) -> None:
    assert_type(sugar_anonymous(x), Tensor[[3]])
    assert_type(native_anonymous(x), Tensor[[3]])


@static_jaxtyping("c")
def sugar_ellipsis(x: Float[Tensor, "... c"]) -> Float[Tensor, "c"]: ...  # noqa: F821


def native_ellipsis[C: IntVar](x: Tensor[[*Elements[IntTuple], C]]) -> Tensor[[C]]: ...


def check_ellipsis(x: Tensor[[2, 3, 4]]) -> None:
    assert_type(sugar_ellipsis(x), Tensor[[4]])
    assert_type(native_ellipsis(x), Tensor[[4]])


@static_jaxtyping("")
def sugar_ellipsis_only(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]: ...


def native_ellipsis_only(x: Tensor[IntTuple]) -> Tensor[IntTuple]: ...


def check_ellipsis_only(x: Tensor[[2, 3]]) -> None:
    assert_type(sugar_ellipsis_only(x), Tensor)
    assert_type(native_ellipsis_only(x), Tensor)


@static_jaxtyping("batch")
def sugar_ellipsis_prefix(
    x: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch"]: ...  # noqa: F821


def native_ellipsis_prefix[B: IntVar](
    x: Tensor[[B, *Elements[IntTuple]]],
) -> Tensor[[B]]: ...


def check_ellipsis_prefix(x: Tensor[[2, 3, 4]]) -> None:
    assert_type(sugar_ellipsis_prefix(x), Tensor[[2]])
    assert_type(native_ellipsis_prefix(x), Tensor[[2]])


@static_jaxtyping("batch channels")
def sugar_ellipsis_both(
    x: Float[Tensor, "batch ... channels"],
) -> Float[Tensor, "channels batch"]: ...


def native_ellipsis_both[B: IntVar, C: IntVar](
    x: Tensor[[B, *Elements[IntTuple], C]],
) -> Tensor[[C, B]]: ...


def check_ellipsis_both(x: Tensor[[2, 3, 4, 5]]) -> None:
    assert_type(sugar_ellipsis_both(x), Tensor[[5, 2]])
    assert_type(native_ellipsis_both(x), Tensor[[5, 2]])


# --- Arithmetic where both operands are named ---


@static_jaxtyping("a b")
def sugar_named_arithmetic(
    x: Float[Tensor, "a"],  # noqa: F821
    y: Float[Tensor, "b"],  # noqa: F821
) -> Float[Tensor, "a+b"]: ...  # noqa: F821


def native_named_arithmetic[A: IntVar, B: IntVar](
    x: Tensor[[A]], y: Tensor[[B]]
) -> Tensor[[A + B]]: ...


def check_named_arithmetic(x: Tensor[[2]], y: Tensor[[3]]) -> None:
    assert_type(sugar_named_arithmetic(x, y), Tensor[[5]])
    assert_type(native_named_arithmetic(x, y), Tensor[[5]])
