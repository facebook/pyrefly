# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar
from typing import overload

T = TypeVar("T")


def transform(x: int, s: str) -> bool:
    return bool(x) and bool(s)


def variadic(*args: int) -> str:
    return ""


# `Type::Forall` / `Forallable::Function`: a generic free function. Its params
# carry a type variable, so they're elided to `...`; the RETURN is concrete
# (`int`) so no TypeVar leaks into the stub.
def generic_fn(x: T) -> int:
    return 0


# `Type::Overload`: an unbound overloaded function whose overloads all share the
# same return type, so it renders as `Callable[..., bytes]`.
@overload
def same_return(x: int) -> bytes: ...
@overload
def same_return(x: str) -> bytes: ...
def same_return(x):
    return b""


# `Type::Overload`: an unbound overloaded function whose overloads have
# DIFFERENT return types, so there is no single faithful return type and it
# falls back to `Incomplete`.
@overload
def diff_return(x: int) -> int: ...
@overload
def diff_return(x: str) -> str: ...
def diff_return(x):
    return x


class C:
    # `Type::BoundMethod` / `BoundMethodType::Function`: a plain instance method.
    # The bound `self` is stripped, leaving `Callable[[int], str]`.
    def method(self, x: int) -> str:
        return ""

    # `Type::BoundMethod` / `BoundMethodType::Forall`: a generic instance method.
    # Params can't be faithfully expressed, so it elides to `Callable[..., int]`.
    def generic_method(self, x: T) -> int:
        return 0

    # `Type::BoundMethod` / `BoundMethodType::Overload`: an overloaded instance
    # method (overloads share a return type → `Callable[..., bool]`).
    @overload
    def overloaded_method(self, x: int) -> bool: ...
    @overload
    def overloaded_method(self, x: str) -> bool: ...
    def overloaded_method(self, x):
        return True


# `Type::Function`: module-level function-valued variables. Their inferred types
# are callables rendered as valid `typing.Callable`, not pyrefly's internal
# `(args) -> ret` display form.
handler = transform
collect = variadic

# `Type::Callable`: a lambda's inferred type is a bare `Callable`.
to_text = lambda n: str(n)

# `Type::Forall` / `Forallable::Function`: reference to the generic free function.
gen_ref = generic_fn

# `Type::Overload` (same returns → `Callable[..., bytes]`).
ov_same = same_return

# `Type::Overload` (differing returns → `Incomplete`).
ov_diff = diff_return

# `Type::BoundMethod` / `BoundMethodType::Function`.
bound_method = C().method

# `Type::BoundMethod` / `BoundMethodType::Forall`.
bound_generic = C().generic_method

# `Type::BoundMethod` / `BoundMethodType::Overload`.
bound_overloaded = C().overloaded_method

# `Type::Forall` / `Forallable::Callable` and `Forallable::TypeAlias` are not
# reachable from value annotations here: a generic *callable* value would
# require a TypeVar in the output (which we deliberately avoid), and a generic
# type alias is a type-level construct, not the inferred type of a value. The
# `Forallable::TypeAlias` arm returns `None` and falls through to the normal
# type display path.
