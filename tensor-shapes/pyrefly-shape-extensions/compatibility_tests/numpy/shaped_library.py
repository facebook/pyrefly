# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Ruff recognizes `Annotated` only under that name, so it reads the shape
# strings of `Shaped` as forward references to undefined names.
# flake8: noqa: F821

"""A library annotated with `Shaped`, as a third-party author would write it.

Pyrefly reads the shape strings, and other checkers ignore them. No checker
should report anything here.

The carrier form, `np.ndarray[Shaped[tuple[int, int], "[M, N]"], dt]`, leaves
other checkers the rank. The shorthand, `Shaped[np.ndarray, "[M, N]"]`, leaves
them a bare array.

The names used inside the shape strings, such as `Elements` and `broadcast`, are
imported. Pyrefly needs them to resolve the strings, and Pyright reports a call
to an undefined name inside the string of an `Annotated`.
"""

import numpy as np
from shape_extensions import broadcast, Elements, shape_vars, Shaped


@shape_vars("M, N")
def transpose(
    x: np.ndarray[Shaped[tuple[int, int], "[M, N]"], np.dtype[np.float64]],
) -> np.ndarray[Shaped[tuple[int, int], "[N, M]"], np.dtype[np.float64]]:
    raise NotImplementedError


@shape_vars("M, N, K")
def matmul(
    a: np.ndarray[Shaped[tuple[int, int], "[M, K]"], np.dtype[np.float64]],
    b: np.ndarray[Shaped[tuple[int, int], "[K, N]"], np.dtype[np.float64]],
) -> np.ndarray[Shaped[tuple[int, int], "[M, N]"], np.dtype[np.float64]]:
    raise NotImplementedError


@shape_vars("N")
def pad_one(
    x: np.ndarray[Shaped[tuple[int], "[N]"], np.dtype[np.float64]],
) -> np.ndarray[Shaped[tuple[int], "[N + 1]"], np.dtype[np.float64]]:
    """Arithmetic in a dimension, which standard type syntax cannot express."""

    raise NotImplementedError


@shape_vars("*Batch, M, N")
def batched_transpose(
    x: np.ndarray[
        Shaped[tuple[int, ...], "[*Elements[Batch], M, N]"], np.dtype[np.float64]
    ],
) -> np.ndarray[
    Shaped[tuple[int, ...], "[*Elements[Batch], N, M]"], np.dtype[np.float64]
]:
    """A variadic splice, which standard type syntax cannot express."""

    raise NotImplementedError


@shape_vars("*S0, *S1")
def add(
    a: np.ndarray[Shaped[tuple[int, ...], "[*Elements[S0]]"], np.dtype[np.float64]],
    b: np.ndarray[Shaped[tuple[int, ...], "[*Elements[S1]]"], np.dtype[np.float64]],
) -> np.ndarray[Shaped[tuple[int, ...], "broadcast(S0, S1)"], np.dtype[np.float64]]:
    """A type-level DSL call, which standard type syntax cannot express."""

    raise NotImplementedError


@shape_vars("M, N")
def untyped_dtype(x: Shaped[np.ndarray, "[M, N]"]) -> Shaped[np.ndarray, "[N, M]"]:
    """The shorthand, for code that never names a dtype."""

    raise NotImplementedError


@shape_vars("Dim, Hidden")
class Encoder:
    """A class-scoped declaration shared by attributes and methods."""

    weight: np.ndarray[Shaped[tuple[int, int], "[Hidden, Dim]"], np.dtype[np.float64]]

    def __init__(
        self,
        weight: np.ndarray[
            Shaped[tuple[int, int], "[Hidden, Dim]"], np.dtype[np.float64]
        ],
    ) -> None:
        self.weight = weight

    def encode(
        self, x: np.ndarray[Shaped[tuple[int], "[Dim]"], np.dtype[np.float64]]
    ) -> np.ndarray[Shaped[tuple[int], "[Hidden]"], np.dtype[np.float64]]:
        raise NotImplementedError


@shape_vars("Dim, Hidden")
def output_size(encoder: Shaped[Encoder, "Dim, Hidden"]) -> Shaped[int, "Hidden"]:
    """A class's declared dimensions, and an integer that is a dimension."""

    raise NotImplementedError


@shape_vars("Dim")
class Stage[T]:
    """A generic class whose declared dimension its consumers may leave out."""

    label: T
    weight: np.ndarray[Shaped[tuple[int], "[Dim]"], np.dtype[np.float64]]
