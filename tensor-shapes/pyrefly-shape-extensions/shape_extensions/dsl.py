# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""DSL internals for shape typing.

Only used inside DSL definition files (e.g. torch/_shapes.pyi), not in
normal stubs or user code.
"""

from __future__ import annotations

import typing

from . import (
    Int as _IntSchema,
    IntTuple as _IntTupleSchema,
    IntTuples as _IntTuplesSchema,
)


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


def Invalid(message: str) -> typing.Any:
    """Return an invalid shape computation from a type-shape DSL body."""

    ...


class Int:
    """Operations that produce values in the DSL integer domain."""

    @staticmethod
    def gradual() -> typing.Any: ...


class IntTuple(_IntTupleSchema):
    """Operations that produce values in the DSL integer-tuple domain."""

    def __new__(cls, values: typing.Iterable[typing.Any]) -> _IntTupleSchema:
        return _IntTupleSchema(values)

    @staticmethod
    def gradual() -> typing.Any: ...


class IntTuples(_IntTuplesSchema):
    """Operations that produce tuples of `IntTuple` values in the DSL."""

    def __new__(cls, values: tuple[_IntTupleSchema, ...]) -> _IntTuplesSchema:
        return _IntTuplesSchema(values)

    @staticmethod
    def gradual() -> typing.Any: ...


# `TypeGuard`, not `TypeIs`: a false result does not narrow symbolic or gradual `Int` values.
def is_concrete_int(value: object) -> typing.TypeGuard[_IntSchema]: ...


def is_int_value(value: object) -> typing.TypeIs[int]: ...


def concat(
    left: typing.Iterable[typing.Any], right: typing.Iterable[typing.Any], /
) -> _IntTupleSchema:
    """Concatenate two shape values inside a type-shape DSL body."""

    return _IntTupleSchema((*left, *right))


@typing.overload
def prod(xs: _IntTupleSchema, /) -> _IntSchema: ...


@typing.overload
def prod(xs: list[int]) -> int: ...


def prod(xs: _IntTupleSchema | list[int]) -> _IntSchema | int:
    """Compute the product of a list of dimension sizes."""
    ...


@typing.overload
def sum(xs: _IntTupleSchema, /) -> _IntSchema: ...


@typing.overload
def sum(xs: list[int]) -> int: ...


def sum(xs: _IntTupleSchema | list[int]) -> _IntSchema | int:
    """Compute the sum of IntTuple dimensions or legacy list integer values."""
    ...


def einsum(spec: str, shapes: _IntTuplesSchema, /) -> _IntTupleSchema:
    """Compute the output shape described by an explicit einsum equation."""
    ...


def parse_einsum_equation(spec: str) -> list[list[list[int]]]:
    """Parse an einsum equation into output locations, equality pairs, and input ranks."""
    ...
