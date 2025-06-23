/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_simple_function_decorator,
    r#"
from typing import assert_type, Callable, Any

def decorator(f: Callable[[int], int]) -> int: ...

@decorator
def decorated(x: int) -> int:
   return x

assert_type(decorated, int)
    "#,
);

testcase!(
    test_abstract_method_implicit_return,
    r#"
import abc
class Foo:
    @abc.abstractmethod
    def foo(self) -> str:
        """
        Some docstring
        """
    "#,
);

testcase!(
    test_identity_function_decorator,
    r#"
from typing import Any, Callable, reveal_type

def decorator[T](f: T) -> T: ...

@decorator
def decorated(x: int) -> str:
   return f"{x}"

reveal_type(decorated)  # E: revealed type: (x: int) -> str
    "#,
);

testcase!(
    test_signature_modifying_function_decorator,
    r#"
from typing import assert_type, Callable, Any

def decorator[T, R](f: Callable[[T], R]) -> Callable[[T, T], R]: ...

@decorator
def decorated(x: int) -> str:
   return f"{x}"

assert_type(decorated, Callable[[int, int], str])
    "#,
);

testcase!(
    test_chaining_decorators,
    r#"
from typing import assert_type, Callable, Any

def decorator0[T, R](f: Callable[[T], R]) -> Callable[[T], set[R]]: ...

def decorator1[T, R](f: Callable[[T], R]) -> Callable[[T], list[R]]: ...

@decorator1
@decorator0
def decorated(x: int) -> str:
   return f"{x}"

assert_type(decorated, Callable[[int], list[set[str]]])
    "#,
);

testcase!(
    test_callable_instance,
    r#"
from typing import Callable, reveal_type
class im_callable:
    def __call__[T](self, arg: T, /) -> T: ...
@im_callable()
def f(x: int) -> int:
    return x
reveal_type(f)  # E: revealed type: (x: int) -> int
    "#,
);

// This test case does not directly use a decorator, but it verifies our
// handling of the `@final` decorator applied to `typing.TypeVar`, which can
// trigger recursion that breaks legacy type parameter handling if we are not
// careful.
testcase!(
    test_that_final_decorator_on_type_var_works,
    r#"
from typing import MutableSequence
x: MutableSequence[int]
    "#,
);

// A regression test for a bug where we were not correctly handling the anywhere
// type for a decorated function.
testcase!(
    test_decorator_general_type,
    r#"
from typing import assert_type, Callable

def decorator(f: Callable[[int], int]) -> int: ...

def anywhere():
    assert_type(decorated, int)

@decorator
def decorated(x: int) -> int:
   return x
    "#,
);

testcase!(
    test_classmethod_first_param,
    r#"
from typing import assert_type

class C:
    @classmethod
    def f(cls) -> int:
        return cls.g()

    @classmethod
    def g(cls) -> int:
        return 42

assert_type(C.f(), int)
assert_type(C.g(), int)
    "#,
);

testcase!(
    test_staticmethod_first_param,
    r#"
from typing import assert_type, Any

class C:
    @staticmethod
    def f(x):
        assert_type(x, Any)

    @staticmethod
    def g(x: int):
        return x

C.f(0)
assert_type(C.g(0), int)
    "#,
);

testcase!(
    test_final,
    r#"
from typing import final, reveal_type
@final
def f(x: int) -> int:
    return x
reveal_type(f)  # E: revealed type: (x: int) -> int
    "#,
);

testcase!(
    test_callable_class_as_decorator,
    r#"
import dataclasses
from typing import Callable, assert_type

@dataclasses.dataclass(frozen=True)
class decorator:
    metadata: int

    def __call__[TReturn, **TParams](
        self, func: Callable[TParams, TReturn]
    ) -> Callable[TParams, TReturn]:
        ...


class C:
    @decorator(42)
    def f(self, x: int) -> int:
        return x

assert_type(C().f(42), int)
    "#,
);

testcase!(
    test_decorate_to_any,
    r#"
from typing import Any, assert_type

def decorate(f) -> Any: ...

class C:
    @decorate
    def f(self, x: int): ...

# `f` is `Any` now, we should be able to call it with anything and get back `Any`.
assert_type(C().f("any", b"thing"), Any)
    "#,
);

testcase!(
    test_decorate_to_generic_callable,
    r#"
from typing import Any, Callable, TypeVar, assert_type
T = TypeVar('T')

def decorate(f) -> Callable[[Any, T], T]:
    return lambda _, x: x

class C:
    @decorate
    def f(self): ...

assert_type(C().f(0), Any)
    "#,
);

testcase!(
    test_decorate_generic_function,
    r#"
from typing import assert_type

def decorate[T](f: T) -> T:
    return f

class C:
    @decorate
    def f[T](self, x: T) -> T:
        return x

assert_type(C().f(0), int)
    "#,
);

testcase!(
    test_property_decorated_to_callback_protocol,
    r#"
from typing import assert_type, Protocol, Callable

class P[**TParams, TReturn](Protocol):
    def __call__(self, *args: TParams.args, **kwargs: TParams.kwargs) -> TReturn: ...

def f[**TParams, TReturn](func: Callable[TParams, TReturn]) -> P[TParams, TReturn]:
    ...

class Foo:
    @property
    @f
    def p(self) -> int:
        return 42

def test(x: Foo) -> None:
    assert_type(x.p, int)
    "#,
);

testcase!(
    test_method_decorated_to_callback_protocol,
    r#"
from typing import assert_type, Protocol, Callable

class P[**TParams, TReturn](Protocol):
    def __call__(self, *args: TParams.args, **kwargs: TParams.kwargs) -> TReturn: ...

def f[**TParams, TReturn](func: Callable[TParams, TReturn]) -> P[TParams, TReturn]:
    ...

class Foo:
    @f
    def p(self) -> int:
        return 42

def test(x: Foo) -> None:
    assert_type(x.p(), int)
    "#,
);

testcase!(
    test_method_decorated_to_callable_instance,
    r#"
class A:
    def __call__(self):
        pass
def decorate(f) -> A:
    return A()
@decorate
def f():
    pass
def g(a: A):
    pass
g(f)
    "#,
);

testcase!(
    bug = "This error message is confusing, I think we need to be clearer when we are printing the *type* of an argument",
    test_decorator_error_message,
    r#"
from typing import Callable, Any
def dec(arg: Callable[..., Any]) -> Callable[..., int]: ...
@dec
def f0(arg: Callable[..., int]) -> Callable[..., int]: ...
@dec  # E: Argument `int` is not assignable to parameter `arg` with type `(...) -> Any` in function `dec`
@f0
def f0(arg: Callable[..., int]) -> Callable[..., int]: ...
    "#,
);

// Reported in https://github.com/facebook/pyrefly/issues/491
testcase!(
    test_total_ordering,
    r#"
from functools import total_ordering
from typing import reveal_type

@total_ordering
class A:
    def __init__(self, x: int) -> None:
        self.x = x
    def __eq__(self, other: "A") -> bool:
        return self.x == other.x
    def __lt__(self, other: "A") -> bool:
        return self.x < other.x

a = A(x=1)
b = A(x=2)

# This should give the correct type for the method `__lt__`
reveal_type(A.__lt__)  # E: revealed type: (self: Self@A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering`
reveal_type(A.__gt__)  # E: revealed type: (self: Self@A, other: A) -> bool
a <= b
"#,
);

testcase!(
    test_total_ordering_no_rich_cmp,
    r#"
from functools import total_ordering

@total_ordering  # E: Class `A` must define at least one of the rich comparison methods.
class A:
    def __init__(self, x: int) -> None:
        self.x = x
"#,
);

testcase!(
    test_total_ordering_dataclass,
    r#"
from dataclasses import dataclass
from functools import total_ordering
from typing import reveal_type

@dataclass
@total_ordering
class A:
    x: int
    def __lt__(self, other: "A") -> bool:
        return self.x < other.x

a = A(x=1)
b = A(x=2)

# This should give the correct type for the method `__lt__`
reveal_type(A.__lt__)  # E: revealed type: (self: Self@A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering`
reveal_type(A.__gt__)  # E: revealed type: (self: Self@A, other: A) -> bool
a <= b
"#,
);

testcase!(
    test_total_ordering_precedence,
    r#"
from functools import total_ordering
from typing import reveal_type

@total_ordering
class A:
    def __init__(self, x: int) -> None:
        self.x = x
    def __eq__(self, other: "A") -> bool:
        return self.x == other.x
    def __lt__(self, other: "A") -> bool:
        return self.x < other.x
    def __le__(self, other: object) -> bool:
        if not isinstance(other, A):
            return NotImplemented
        return self.x <= other.x

# This should give the correct type for the method `__lt__`
reveal_type(A.__lt__)  # E: revealed type: (self: Self@A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering` via `__lt__`
reveal_type(A.__gt__)  # E: revealed type: (self: Self@A, other: A) -> bool

# This should give the correct type for the method `__le__`
reveal_type(A.__le__)  # E: revealed type: (self: Self@A, other: object) -> bool
# This should give be synthesized via `functools.total_ordering` via `__le__`
reveal_type(A.__ge__)  # E: revealed type: (self: Self@A, other: object) -> bool
"#,
);
