# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""DSL internals for shape typing.

Only used inside DSL definition files (e.g. torch/_shapes.pyi), not in
normal stubs or user code.
"""

# TODO(stroxler): Unquote `typing.TypeIs` when Python 3.13+ is the minimum version.
import typing

from . import (
    Int as _IntSchema,
    IntTuple as _IntTupleSchema,
    IntTuples as _IntTuplesSchema,
)

# DSL builtins: these exist so that DSL definition files can import them
# and avoid unbound-name errors. The DSL compiler recognizes these names
# as builtins regardless of the Python-level definitions here.


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

    def __new__(cls, values: typing.Iterable[_IntTupleSchema]) -> _IntTuplesSchema:
        return _IntTuplesSchema(values)

    @staticmethod
    def gradual() -> typing.Any: ...


# `TypeGuard`, not `TypeIs`: a false result does not narrow symbolic or gradual `Int` values.
def is_concrete_int(value: object) -> typing.TypeGuard[_IntSchema]: ...


def is_int_value(value: object) -> "typing.TypeIs[int]": ...


def concat(
    left: typing.Iterable[typing.Any], right: typing.Iterable[typing.Any], /
) -> _IntTupleSchema:
    """Concatenate two shape values inside a type-shape DSL body."""

    return _IntTupleSchema((*left, *right))


def prod(xs: _IntTupleSchema, /) -> _IntSchema: ...


def sum(xs: _IntTupleSchema, /) -> _IntSchema: ...


def einsum(spec: str, shapes: _IntTuplesSchema, /) -> _IntTupleSchema:
    """Compute the output shape described by an explicit einsum equation."""
    ...


def _gufunc_broadcast(spec: str, shapes: _IntTuplesSchema, /) -> _IntTupleSchema:
    """Compute a gufunc result shape inside a type-level shape DSL function."""
    raise NotImplementedError(
        "dsl._gufunc_broadcast is only available inside a type-level shape DSL function"
    )
