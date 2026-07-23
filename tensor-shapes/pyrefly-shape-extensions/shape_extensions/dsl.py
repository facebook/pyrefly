# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""DSL internals for shape typing.

Only used inside DSL definition files (e.g. torch/_shapes.pyi), not in
normal stubs or user code.
"""

import typing


def shape_dsl_function(fn: typing.Callable) -> typing.Callable:
    """Marks a function as a shape DSL function.

    At runtime this is a no-op: the decorated function is returned unchanged.
    Pyrefly uses this decorator at type-checking time to convert the function
    body to DSL IR via convert_fndef.
    """
    return fn


# DSL builtins: these exist so that DSL definition files can import them
# and avoid unbound-name errors. The DSL compiler recognizes these names
# as builtins regardless of the Python-level definitions here.


class ShapedArray:
    """A shaped-array value in the DSL, constructed via ShapedArray(shape=[...])."""

    shape: list[int]

    def __init__(self, *, shape: list[int]) -> None:
        self.shape = shape


class symint:
    """A symbolic integer dimension in the DSL."""

    ...


class Error(Exception):
    """DSL error raised via `raise Error("message")`."""

    ...


class _Unknown:
    """Sentinel returned from DSL functions to fall back to the declared return type."""

    ...


Unknown: _Unknown = _Unknown()


def prod(xs: list[int]) -> int:
    """Compute the product of a list of dimension sizes."""
    ...


def sum(xs: list[int]) -> int:
    """Compute the sum of a list of dimension sizes."""
    ...


def parse_einsum_equation(spec: str) -> list[list[list[int]]]:
    """Parse an einsum equation string into output map and check pairs."""
    ...
