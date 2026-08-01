# @generated
from typing import Callable
from _typeshed import Incomplete

from typing import overload, TypeVar

T = TypeVar("T")


def transform(x: int, s: str) -> bool: ...


def variadic(*args: int) -> str: ...


def optional_param(x: int, y: int = 0) -> bool: ...


def generic_fn(x: T) -> int: ...


@overload
def same_return(x: int) -> bytes: ...


@overload
def same_return(x: str) -> bytes: ...


@overload
def diff_return(x: int) -> int: ...


@overload
def diff_return(x: str) -> str: ...


class C:
    def method(self, x: int) -> str: ...

    def generic_method(self, x: T) -> int: ...

    @overload
    def overloaded_method(self, x: int) -> bool: ...

    @overload
    def overloaded_method(self, x: str) -> bool: ...


handler: Callable[[int, str], bool]
collect: Callable[..., str]
optional_ref: Callable[..., bool]
to_text: Callable[[Incomplete], str]
gen_ref: Callable[..., int]
ov_same: Callable[..., bytes]
ov_diff: Callable[..., Incomplete]
bound_method: Callable[[int], str]
bound_generic: Callable[..., int]
bound_overloaded: Callable[..., bool]
