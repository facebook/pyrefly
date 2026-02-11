/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
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
    test_parameter_type_inferred_from_decorator,
    r#"
from typing import Callable, reveal_type

def enforce_int_arg(func: Callable[[int], None]) -> Callable[[int], None]:
    return func

@enforce_int_arg
def takes_inferred(i) -> None:
    reveal_type(i)  # E: revealed type: int
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
    test_abstract_classmethod,
    r#"
from typing import assert_type
import abc

class C(abc.ABC):
    @abc.abstractclassmethod
    def f(cls) -> int:
        return 42

assert_type(C.f(), int)
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
    test_abstract_staticmethod,
    r#"
from typing import assert_type
import abc

class C(abc.ABC):
    @abc.abstractstaticmethod
    def f() -> int:
        return 42

assert_type(C.f(), int)
    "#,
);

testcase!(
    test_final,
    r#"
from typing import final, reveal_type
@final  # E: `@final` can only be used on methods
def f(x: int) -> int:
    return x
reveal_type(f)  # E: revealed type: (x: int) -> int
    "#,
);

testcase!(
    test_invalid_top_level_function_decorators,
    r#"
from typing import *
from abc import abstractstaticmethod, abstractmethod # E: `abstractstaticmethod` is deprecated
from enum import member, nonmember

@member  # E: can only be used on methods
@nonmember  # E: can only be used on methods
@abstractmethod  # E: can only be used on methods
@staticmethod  # E: can only be used on methods
@classmethod  # E: can only be used on methods
@abstractstaticmethod  # E: can only be used on methods
@property  # E: can only be used on methods
@final  # E: can only be used on methods
@override  # E: can only be used on methods
def f(x: int) -> int:
    return x
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

assert_type(C().f(0), int)
    "#,
);

// Test that `@decorate` doesn't interfere with reading other decorators, by verifying that an
// override error is still reported.
testcase!(
    test_override_error_with_generic_callable_decorator,
    r#"
from typing import Any, Callable, TypeVar, assert_type, override
T = TypeVar('T')

def decorate(f) -> Callable[[Any, T], T]:
    return lambda _, x: x

class C:
    @override
    @decorate
    def f(self): ...  # E: no parent class has a matching attribute

    @decorate
    @override
    def g(self): ...  # E: no parent class has a matching attribute
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
from typing import assert_type, Protocol, Callable, Concatenate, Any

class P[**TParams, TReturn](Protocol):
    def __call__(self, *args: TParams.args, **kwargs: TParams.kwargs) -> TReturn: ...

def f[**TParams, TReturn](func: Callable[Concatenate[Any, TParams], TReturn]) -> P[TParams, TReturn]:
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
from typing import assert_type, Protocol, Callable, Concatenate, Any

class P[**TParams, TReturn](Protocol):
    def __call__(self, *args: TParams.args, **kwargs: TParams.kwargs) -> TReturn: ...

def f[**TParams, TReturn](func: Callable[Concatenate[Any, TParams], TReturn]) -> P[TParams, TReturn]:
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
    def __eq__(self, other: "A") -> bool:  # E: `A.__eq__` overrides parent class `object` in an inconsistent manner
        return self.x == other.x
    def __lt__(self, other: "A") -> bool:
        return self.x < other.x

a = A(x=1)
b = A(x=2)

# This should give the correct type for the method `__lt__`
reveal_type(A.__lt__)  # E: revealed type: (self: A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering`
reveal_type(A.__gt__)  # E: revealed type: (self: A, other: A) -> bool
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
reveal_type(A.__lt__)  # E: revealed type: (self: A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering`
reveal_type(A.__gt__)  # E: revealed type: (self: A, other: A) -> bool
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
    def __eq__(self, other: "A") -> bool:  # E: `A.__eq__` overrides parent class `object` in an inconsistent manner
        return self.x == other.x
    def __lt__(self, other: "A") -> bool:
        return self.x < other.x
    def __le__(self, other: object) -> bool:
        if not isinstance(other, A):
            return NotImplemented
        return self.x <= other.x

# This should give the correct type for the method `__lt__`
reveal_type(A.__lt__)  # E: revealed type: (self: A, other: A) -> bool
# This should give be synthesized via `functools.total_ordering` via `__lt__`
reveal_type(A.__gt__)  # E: revealed type: (self: A, other: A) -> bool

# This should give the correct type for the method `__le__`
reveal_type(A.__le__)  # E: revealed type: (self: A, other: object) -> bool
# This should give be synthesized via `functools.total_ordering` via `__le__`
reveal_type(A.__ge__)  # E: revealed type: (self: A, other: object) -> bool
"#,
);

testcase!(
    test_overload_with_docstring,
    r#"
from typing import overload, Any
@overload
def foo(a: int) -> int: ...
@overload
def foo(a: str) -> str:
    """Docstring"""
def foo(*args, **kwargs) -> Any:
    pass

    "#,
);

testcase!(
    test_overload_with_docstring2,
    r#"
from typing import overload, Any
@overload
def foo(a: int) -> int: ...
@overload
def foo(a: str) -> str:
    """Docstring"""
    return 123             # E: Returned type `Literal[123]` is not assignable to declared return type `str`
def foo(*args, **kwargs) -> Any:
    pass

    "#,
);

testcase!(
    test_abstract_method_skip_return,
    r#"
from abc import abstractmethod

class C:
        @abstractmethod
        def m1(self) -> int:
            return NotImplemented

        @abstractmethod
        def m2(self) -> int:
            pass

        @abstractmethod
        def m3(self) -> int:
            """some docstring"""

        @abstractmethod
        def m4(self) -> int: ...

        @abstractmethod
        def method5(self) -> int:
            return "sub" # E: Returned type `Literal['sub']` is not assignable to declared return type `int`
    "#,
);

testcase!(
    bug = "We should treat `A.f` as a classmethod",
    test_desugared_decorator_application,
    r#"
from typing import assert_type
class A:
    def f(cls):
        return cls
    f = classmethod(f)
assert_type(A.f(), type[A])  # E: assert_type(A, type[A])  # E: `type[A]` is not assignable to parameter `cls` with type `A`
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/793 - we don't
// want the `contextmanager` decorator to lose the "forall-ness" of the decorated
// function.
testcase!(
    test_contextmanager_preserves_tparams_and_forall,
    r#"
import contextlib
from contextlib import contextmanager
from typing import Iterator, List, assert_type

@contextmanager
def generic_ctx[T](val: T) -> Iterator[T]:
    yield val

def test(x: int, items: List[int]):
    with generic_ctx(x) as val:
        assert_type(val, int)
    m = generic_ctx(items)
    assert_type(m, contextlib._GeneratorContextManager[List[int], None, None])
"#,
);

// When we preserve "forall-ness" of decorated callables, we need to make sure
// that we don't accidentally leak `Type::Quantified` when a decorator returns
// a generic non-callable type.
testcase!(
    test_decorator_returns_non_callable_produces_gradual_type,
    r#"
from typing import TypeVar, Callable, List, assert_type, Any

def list_decorator[T](f: Callable[[T], T]) -> List[T]: ...

@list_decorator
def my_func[T](x: T) -> T:
    return x

def test():
    assert_type(my_func, List[Any])
"#,
);

// If a decorator returns a callable that has additional tparams that weren't
// in the original function, we need to make sure the decorated function type is
// generic over all tparams.
testcase!(
    test_decorator_adds_new_tparams_to_forall,
    r#"
from typing import TypeVar, Callable, Tuple, assert_type

def add_generic[T, S](f: Callable[[T], T]) -> Callable[[T, S], Tuple[T, S]]:
    return lambda x, y: (f(x), y)

@add_generic
def my_func[T](x: T) -> T:
    return x

def test(x: int, y: str):
    res = my_func(x, y)
    assert_type(res, Tuple[int, str])
"#,
);

// If a decorator converts a generic callable into a non-generic, we should drop the callable's tparams
testcase!(
    test_decorator_strips_tparams_and_forall,
    r#"
from typing import TypeVar, Callable, Tuple, reveal_type

def add_generic[T](f: Callable[[T], T]) -> Callable[[object], None]:
    ...

@add_generic
def my_func[T](x: T) -> T:
    return x

reveal_type(my_func)  # E: revealed type: (object) -> None
"#,
);

fn env_numba() -> TestEnv {
    let mut env = TestEnv::one_with_path(
        "numba",
        "numba/__init__.pyi",
        r#"
from numba.core.decorators import jit, njit
"#,
    );
    env.add_with_path(
        "numba.core.decorators",
        "numba/core/decorators.pyi",
        r#"
def jit(*args, **kwargs): ...
def njit(*args, **kwargs): ...
"#,
    );
    env
}

testcase!(
    test_numba_jit_decorators_preserve_signature,
    env_numba(),
    r#"
import numba
from typing import assert_type

@numba.jit(nopython=True)
def test1(a: int, b: int) -> int:
    return a + b

@numba.njit(cache=True)
def test2(a: int, b: int) -> int:
    return a + b

assert_type(test1(1, 2), int)
assert_type(test2(1, 2), int)
"#,
);
