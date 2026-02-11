/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_missing_return,
    r#"

def f() -> int:  # E: Function declared to return `int` but is missing an explicit `return`
    pass
"#,
);

testcase!(
    test_missing_return_none,
    r#"
def f() -> None:
    pass
"#,
);

testcase!(
    test_missing_return_implicit,
    r#"
from typing import assert_type

def f():
    pass
assert_type(f(), None)
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1491
testcase!(
    test_infer_return_in_for_loop,
    r#"
from typing import reveal_type

class A:
    def f(self, x):
        for y in x:
            pass

reveal_type(A().f(0))  # E: revealed type: None
"#,
);

testcase!(
    test_return_unions,
    r#"
from typing import assert_type, Literal

def f(b: bool):
    if b:
        return 1
    else:
        return "test"
assert_type(f(True), Literal['test', 1])
"#,
);

testcase!(
    test_return_some_return,
    r#"
from typing import assert_type

def f(b: bool) -> int:  # E: Function declared to return `int`, but one or more paths are missing an explicit `return`
    if b:
        return 1
    else:
        pass
"#,
);

testcase!(
    test_return_catch,
    r#"
def f(b: bool) -> int:
    try:
        return 1
    except Exception:
        return 2
"#,
);

testcase!(
    test_return_never,
    r#"
from typing import NoReturn

def fail() -> NoReturn:
    raise Exception()

def f(b: bool) -> int:
    if b:
        return 1
    else:
        fail()
"#,
);

testcase!(
    test_return_never_should_not_fail,
    r#"
from typing import NoReturn

def fail() -> NoReturn:
    raise Exception()

def f() -> int:
   fail()
"#,
);

testcase!(
    test_return_none_should_fail,
    r#"
def does_not_fail() -> None:
    return None

def f(b: bool) -> int: # E: Function declared to return `int`, but one or more paths are missing an explicit `return`
    if b:
        return 1
    else:
        does_not_fail()
"#,
);

testcase!(
    test_return_should_fail,
    r#"
def fail():
    pass

def f() -> int: # E: Function declared to return `int` but is missing an explicit `return`
   fail()
"#,
);

testcase!(
    test_return_if_no_else_real,
    r#"
def f(b: bool) -> int:  # E: Function declared to return `int`, but one or more paths are missing an explicit `return`
    if b:
        return 1
"#,
);

testcase!(
    test_return_constrained_typevar_isinstance,
    r#"
from typing import TypeVar

T = TypeVar("T", int, str)

def f(x: T) -> T:
    if isinstance(x, int):
        return x
    elif isinstance(x, str):
        return x
"#,
);

testcase!(
    test_return_if_no_else_none,
    r#"
def f(b: bool) -> None:
    if b:
        return None
"#,
);

testcase!(
    test_return_then_dead_code,
    r#"
def f(b: bool) -> int:  # E: Function declared to return `int`, but one or more paths are missing an explicit `return`
    return 1
    # This code is unreachable. A linter should spot this.
    # But for now, it's perfectly reasonable to say the `pass`
    # has the wrong type, and a `return` should be here.
    pass
"#,
);

testcase!(
    test_infer_never,
    r#"
from typing import assert_type, Never

def f():
    raise Exception()

assert_type(f(), Never)
"#,
);

testcase!(
    test_infer_never2,
    r#"
from typing import NoReturn, assert_type, Literal

def fail() -> NoReturn:
    raise Exception()

def f(b: bool):
    if b:
        return 1
    else:
        fail()

assert_type(f(True), Literal[1])
"#,
);

testcase!(
    test_infer_never3,
    r#"
from typing import assert_type

def f() -> int:
   raise Exception()
assert_type(f(), int)
"#,
);

testcase!(
    test_return_never_with_unreachable,
    r#"
from typing import NoReturn

def fail() -> NoReturn:
    raise Exception()

def f(b: bool) -> int:
    if b:
        return 1
    else:
        fail()
        return 4
"#,
);

testcase!(
    bug = "conformance: Unreachable code after NoReturn should not be type-checked",
    test_return_never_unreachable_wrong_type,
    r#"
from typing import NoReturn

def stop() -> NoReturn:
    raise RuntimeError("stop")

def f(x: int) -> int:
    if x > 0:
        return x
    stop()
    return "wrong type"  # E: Returned type `Literal['wrong type']` is not assignable to declared return type `int`
"#,
);

testcase!(
    test_return_never_error_return,
    r#"
def f(x: int): pass

def g():
   return f("test") # E: Argument `Literal['test']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_return_no_error,
    r#"
def B() -> None:
    (3)
"#,
);

testcase!(
    test_return_never_with_wrong_type,
    r#"
from typing import NoReturn

def fail() -> NoReturn:
    raise Exception()

def f(b: bool) -> int:
    if b:
        return None # E: Returned type `None` is not assignable to declared return type `int`
    else:
        fail()
"#,
);

testcase!(
    test_return_error_on_docstring,
    r#"
def f() -> int: # E: Function declared to return `int` but is missing an explicit `return`
    """ ... """
     "#,
);

testcase!(
    test_async_return_inference,
    r#"
from typing import assert_type, Any, Callable, Coroutine
x: int = ...  # E:
async def async_f_annotated() -> int:
    return x
async def async_f_inferred():
    return x
assert_type(async_f_annotated, Callable[[], Coroutine[Any, Any, int]])
assert_type(async_f_inferred, Callable[[], Coroutine[Any, Any, int]])
     "#,
);

testcase!(
    test_toplevel_return_empty,
    r#"
return # E: Invalid `return` outside of a function
"#,
);

testcase!(
    test_toplevel_return_expr,
    r#"
def f(x: str): pass

return f(1) # E: Invalid `return` outside of a function # E: `Literal[1]` is not assignable to parameter `x` with type `str`
"#,
);

testcase!(
    test_bare_return_with_non_none_type,
    r#"
def test() -> int:
    return  # E: Returned type `None` is not assignable to declared return type `int`
"#,
);

testcase!(
    test_bare_return_with_none_type,
    r#"
def test() -> None:
    return  # Should work - None is assignable to None
"#,
);

testcase!(
    test_bare_return_in_generator,
    r#"
from typing import Generator

def gen() -> Generator[int, None, str]:
    yield 1
    return  # E: Returned type `None` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_unreachable_return_after_return,
    r#"
def test() -> int:
    return 1
    # values in unreachable returns do not get checked against the annotation
    return "" # E: This `return` statement is unreachable
"#,
);

testcase!(
    test_unreachable_return_after_raise,
    r#"
def test():
    raise Exception()
    return 1 # E: This `return` statement is unreachable
"#,
);

testcase!(
    test_unreachable_yield_after_return,
    r#"
def test():
    return 1
    yield 2 # E: This `yield` expression is unreachable
"#,
);

testcase!(
    test_unreachable_return_after_break,
    r#"
def test():
    while True:
        break
        return 1 # E: This `return` statement is unreachable
"#,
);

testcase!(
    test_unreachable_return_after_continue,
    r#"
def test():
    while True:
        continue
        return 1 # E: This `return` statement is unreachable
"#,
);

// Context managers may swallow exceptions, so we cannot guarantee that execution does not continue
testcase!(
    test_unreachable_return_after_error_swallowing_context_manager,
    r#"
from contextlib import suppress
def is_zero(x: int):
    with suppress(ZeroDivisionError):
        1 / x
        return False
    return True
"#,
);

// We shouldn't flag an unreachable return in the else branch of a static check
testcase!(
    test_unreachable_return_after_static_check,
    r#"
import sys
def test():
    if sys.version_info >= (3, 8):
        return True
    return False
"#,
);

testcase!(
    test_yield_after_yield_is_ok,
    r#"
def test():
    yield 1
    yield 2  # No error - yields can follow other yields
"#,
);

testcase!(
    test_unreachable_yield_from_after_return,
    r#"
def test():
    return 1
    yield from [2, 3] # E: This `yield from` expression is unreachable
"#,
);

testcase!(
    test_no_missing_return_for_stubs,
    r#"
from typing import Protocol, overload
from abc import abstractmethod

class P(Protocol):
    def f1(self) -> int:
        """a"""
    def f2(self) -> int:
        """a"""
        ...
    def f3(self) -> int:
        """a"""
        pass
    def f4(self) -> int:
        """a"""
        return NotImplemented
    def f5(self) -> int:
        """a"""
        raise NotImplementedError()
    def f6(self) -> int:
        ...
    def f7(self) -> int:
        pass
    def f8(self) -> int:
        return NotImplemented
    def f9(self) -> int:
        raise NotImplementedError()

class C:
    def f1(self) -> int:  # E:
        """a"""
    def f2(self) -> int:
        """a"""
        ...  # OK - other type checkers do not permit this outside of stub files
    def f3(self) -> int:  # E:
        """a"""
        pass
    def f4(self) -> int:
        """a"""
        return NotImplemented  # OK
    def f5(self) -> int:
        """a"""
        raise NotImplementedError()  # OK
    def f6(self) -> int:
        ...  # OK - other type checkers do not permit this outside of stub files
    def f7(self) -> int:  # E:
        pass
    def f8(self) -> int:
        return NotImplemented  # OK
    def f9(self) -> int:
        raise NotImplementedError()  # OK

class AbstractC:
    @abstractmethod
    def f1(self) -> int:
        """a"""
    @abstractmethod
    def f2(self) -> int:
        """a"""
        ...
    @abstractmethod
    def f3(self) -> int:
        """a"""
        pass
    @abstractmethod
    def f4(self) -> int:
        """a"""
        return NotImplemented
    @abstractmethod
    def f5(self) -> int:
        """a"""
        raise NotImplementedError()
    @abstractmethod
    def f6(self) -> int:
        ...
    @abstractmethod
    def f7(self) -> int:
        pass
    @abstractmethod
    def f8(self) -> int:
        return NotImplemented
    @abstractmethod
    def f9(self) -> int:
        raise NotImplementedError()

class OverloadC:
    @overload
    def f(self) -> int:
        """a"""
    @overload
    def f(self) -> int:
        """a"""
        ...
    @overload
    def f(self) -> int:
        """a"""
        pass
    @overload
    def f(self) -> int:
        """a"""
        return NotImplemented
    @overload
    def f(self) -> int:
        """a"""
        raise NotImplementedError()
    @overload
    def f(self) -> int:
        ...
    @overload
    def f(self) -> int:
        pass
    @overload
    def f(self) -> int:
        return NotImplemented
    @overload
    def f(self) -> int:
        raise NotImplementedError()
    def f(self) -> int:
        return 0
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2141
// List concatenation with contextual return type hint should work
testcase!(
    test_return_list_concat_contextual_hint,
    r#"
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def foo(self, x: int) -> None: ...

class A(Base):
    def foo(self, x: int) -> None:
        print(x)

class B(Base):
    def foo(self, x: int) -> None:
        pass

# This should type-check without error: the return type hint list[Base]
# provides context for inferring [A()] + [B()] as list[Base].
def return_object(name: str) -> list[Base]:
    return [A()] + [B()]

# Non-list-returning variant still works (for comparison)
def return_object_non_list(name: str) -> Base:
    o = None
    if name == "a":
        o = A()
    else:
        o = B()
    return o
"#,
);
