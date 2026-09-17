/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_simple_with,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:
    assert_type(foo, int)

bar: str = "abc"
with Foo() as bar: # E: `int` is not assignable to variable `bar` with type `str`
    assert_type(bar, str)
    "#,
);

testcase!(
    test_simple_async_with,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    async def __aenter__(self) -> int:
        ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo:
        assert_type(foo, int)
    "#,
);

testcase!(
    test_simple_with_error,
    r#"
def test_sync() -> None:
    with 42 as foo:  # E: Cannot use `Literal[42]` as a context manager\n  Object of class `int` has no attribute `__enter__` # E: has no attribute `__exit__`
        pass

async def test_async() -> None:
    async with "abc" as bar:  # E: has no attribute `__aenter__` # E: has no attribute `__aexit__`
        pass
    "#,
);

testcase!(
    test_simple_with_wrong_enter_type,
    r#"
from types import TracebackType
class Foo:
    __enter__: int = 42
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:  # E: Expected `__enter__` to be a callable, got `int`
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_attribute_type,
    r#"
from types import TracebackType
class Foo:
    def __enter__(self) -> int: ...
    __exit__: int = 42

with Foo() as foo:  # E: Expected `__exit__` to be a callable, got `int`
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_argument_count,
    r#"
from typing import assert_type
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(self) -> None:
        ...

with Foo() as foo:  # E: Expected 0 positional arguments, got 3
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_argument_type,
    r#"
from typing import assert_type
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(self, exc_type: int, exc_value: int, traceback: int) -> None:
        ...

with Foo() as foo: # E: `__exit__` must be callable with the argument types (type[BaseException], BaseException, TracebackType) # E: `__exit__` must be callable with the argument types (None, None, None)
    pass
    "#,
);

testcase!(
    test_with_wrong_return_type,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> str:
        ...

with Foo() as foo:  # E: Cannot use `Foo` as a context manager\n  Return type `str` of function `Foo.__exit__` is not assignable to expected return type `bool | None`
    pass
    "#,
);

testcase!(
    test_async_with_dunder_aenter_not_async,
    r#"
from types import TracebackType
class Foo:
    def __aenter__(self) -> int:
        ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo: # E: Expected `__aenter__` to be async
        ...
"#,
);

testcase!(
    test_async_with_dunder_aexit_not_async,
    r#"
from types import TracebackType
class Foo:
    async def __aenter__(self) -> int:
        ...
    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo: # E: Expected `__aexit__` to be async
        ...
"#,
);

testcase!(
    test_with_return_bool,
    r#"
class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> bool:
    return False

def f() -> int:  # E: missing an explicit `return`
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_true,
    r#"
from typing import Literal

class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> Literal[True]:
    return True

def f() -> int:  # E: missing an explicit `return`
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_false,
    r#"
# From https://github.com/facebook/pyrefly/issues/24

from typing import Literal

class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> Literal[False]:
    return False

def f() -> int:
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_any,
    r#"
# From https://github.com/facebook/pyrefly/issues/24

from typing import Any

def f(x: Any) -> int:
  with x:
    return 1
"#,
);

testcase!(
    test_with_contextmanager,
    r#"
import contextlib
from typing import Generator

@contextlib.contextmanager
def f() -> Generator[str, None, None]:
    yield ""

def g() -> bool:
    with f():
        return True
    "#,
);

testcase!(
    test_overloaded_exit_with,
    r#"
from typing import assert_type, overload
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
        /
    ) -> None:
        ...
    @overload
    def __exit__(
        self,
        exc_type: None,
        exc_value: None,
        traceback: None,
        /
    ) -> None:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:
    assert_type(foo, int)
    "#,
);

testcase!(
    test_context_manager_exception_suppression_conformance,
    r#"
from typing import Any, Literal, assert_type

class CMBase:
    def __enter__(self) -> None:
        pass

class Suppress1(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return True

class Suppress2(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Literal[True]:
        return True

class NoSuppress1(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

class NoSuppress2(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        return False

class NoSuppress3(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Any:
        return False

class NoSuppress4(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> None | bool:
        return None

def suppress1(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress1():
            raise ValueError
    assert_type(x, int | str)

def suppress2(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress2():
            raise ValueError
    assert_type(x, int | str)

def no_suppress1(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress1():
            raise ValueError
    assert_type(x, str)

def no_suppress2(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress2():
            raise ValueError
    assert_type(x, str)

def no_suppress3(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress3():
            raise ValueError
    assert_type(x, str)

def no_suppress4(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress4():
            raise ValueError
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_multiple_items,
    r#"
from typing import assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def outer_suppresses(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress(), NoSuppress():
            raise ValueError
    assert_type(x, int | str)

def neither_suppresses(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress(), NoSuppress():
            raise ValueError
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_async,
    r#"
from typing import assert_type

class Suppress:
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, exc_type, exc_value, traceback) -> bool: ...

async def f(x: int | str) -> None:
    if isinstance(x, int):
        async with Suppress():
            raise ValueError
    assert_type(x, int | str)
"#,
);

testcase!(
    test_with_suppression_no_return_call,
    r#"
from typing import NoReturn, assert_type

def fail() -> NoReturn: ...

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def suppressed(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            fail()
    assert_type(x, int | str)

def not_suppressed(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress():
            fail()
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_uninitialized,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def suppressed(c: bool) -> int:
    if c:
        y = 1
    else:
        with Suppress():
            raise ValueError
    return y  # E: `y` may be uninitialized

def not_suppressed(c: bool) -> int:
    if c:
        y = 1
    else:
        with NoSuppress():
            raise ValueError
    return y
"#,
);

// `__exit__` runs for `return`/`break`/`continue`, but its return value is only
// consulted when an exception is in flight, so a suppressing context manager cannot
// cancel them the way it cancels a `raise`.
testcase!(
    test_with_terminators_are_not_suppressible,
    r#"
from typing import assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def ret(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            return
    assert_type(x, str)

def brk(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                break
        assert_type(x, str)

def cont(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                continue
        assert_type(x, str)

def raises(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            raise ValueError
    assert_type(x, int | str)

def sys_exit(x: int | str) -> None:
    import sys
    if isinstance(x, int):
        with Suppress():
            sys.exit(1)
    assert_type(x, int | str)

def os_exit(x: int | str) -> None:
    import os
    if isinstance(x, int):
        with Suppress():
            os._exit(1)
    assert_type(x, str)

def ret_or_raise(c: bool, x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            if c:
                return
            else:
                raise ValueError
    assert_type(x, int | str)

def ret_or_ret(c: bool, x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            if c:
                return
            else:
                return
    assert_type(x, str)

# Entering the inner manager happens inside the outer's extent, so an exception from it can
# be suppressed and the `return` never reached, leaving this branch able to fall through.
def nested_ret(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                return
    assert_type(x, int | str)

def nested_raise(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                raise ValueError
    assert_type(x, int | str)
"#,
);

// An operation before a jump may raise. If the context manager suppresses that
// exception, the jump is never executed and control resumes after the `with`.
testcase!(
    test_with_exception_before_terminator_may_be_suppressed,
    r#"
from typing import TypeVar, assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Exploding:
    def __enter__(self) -> None:
        raise RuntimeError
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def may_raise() -> None: ...

def brk(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                may_raise()
                break
        assert_type(x, int | str)

def cont(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                may_raise()
                continue
        assert_type(x, int | str)

def unreachable_in_with() -> None:
    with Suppress():
        return
        print("dead")  # E: This code is unreachable

def no_suppression(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with NoSuppress():
                may_raise()
                break
        assert_type(x, str)

def fallback_after_suppressed_continue() -> None:
    while True:
        with Suppress():
            may_raise()
            continue
        print("the exception bypassed continue")
        break

def fallback_after_suppressed_break() -> None:
    while True:
        with Suppress():
            may_raise()
            break
        print("the exception bypassed break")

def no_operation_before_continue() -> None:
    while True:
        with Suppress():
            continue
        print("continue always executes")  # E: This code is unreachable

def raise_inside_a_special_export_assignment() -> None:
    # `TypeVar(...)` is bound by an early-returning arm of `stmt`; it may still raise.
    while True:
        with Suppress():
            T = TypeVar("T")
            break
        print("the exception bypassed the break")

def bare_return_always_executes() -> None:
    # A bare `return` evaluates nothing, so it cannot be bypassed, matching `break`.
    with Suppress():
        return
    print("return always executes")  # E: This code is unreachable

def no_operation_before_break() -> None:
    while True:
        with Suppress():
            break
        print("break always executes")  # E: This code is unreachable

def exploding_with() -> None:
    with Suppress(), Exploding():
        return
    print("reachable")

def exception_in_finally() -> None:
    with Suppress():
        try:
            return
        finally:
            may_raise()
    print("reachable")
"#,
);

// A statement that terminates the flow has still evaluated its header by then, so an
// enclosing context manager may suppress an exception from that header and skip the jump.
// Recording the header separately keeps this distinct from a bare `return`/`break`/`continue`,
// which evaluates nothing and stays unsuppressible per
// `test_with_terminators_are_not_suppressible`.
testcase!(
    test_with_exception_in_a_terminating_test_expression,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def may_raise_bool() -> bool: ...

def f() -> None:
    while True:
        with Suppress():
            if may_raise_bool():
                break
            else:
                break
        print("the exception bypassed both breaks")
"#,
);

// Entering a manager after the first can raise and be suppressed by an earlier one, so
// binding leaves the flow reachable after a `with` that enters more than one. The
// `__exit__` types settle whether any of them really suppresses, so the diagnostic is
// deferred to solving.
testcase!(
    test_dead_code_after_multi_manager_with,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def combined() -> None:
    with NoSuppress(), NoSuppress():
        return
    print("dead")  # E: This code is unreachable

def nested() -> None:
    with NoSuppress():
        with NoSuppress():
            return
    print("dead")  # E: This code is unreachable
"#,
);

// One suppressing manager anywhere in the chain is enough to keep the fall-through alive.
testcase!(
    test_live_code_after_multi_manager_with,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def outer_suppresses() -> None:
    with Suppress(), NoSuppress():
        return
    print("reachable")

def inner_suppresses() -> None:
    with NoSuppress(), Suppress():
        return
    print("reachable")
"#,
);

// A `with` whose body exits under a static test must not make the following code dead: the
// exit only happens on other configurations. This is why the check is gated on the definite
// termination flag rather than on `has_terminated`, which a static test also sets.
testcase!(
    test_no_report_after_with_exited_by_static_test,
    r#"
import sys

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def asserted() -> None:
    with NoSuppress():
        assert sys.version_info >= (3, 20)
    print("runs on a new enough Python")

def gated_return() -> None:
    with NoSuppress(), NoSuppress():
        if sys.version_info < (3, 20):
            return
    print("runs on a new enough Python")
"#,
);

// Claiming code is dead requires knowing that no manager suppresses, which is stronger than
// failing to prove that one does. A manager we cannot read might suppress at runtime.
testcase!(
    test_no_report_after_with_when_suppression_is_unknown,
    r#"
import contextlib
from typing import Any

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def anything() -> Any: ...

def gradual() -> None:
    with anything():
        raise ValueError()
    print("an `Any` manager might suppress")

def bool_or_none() -> None:
    with contextlib.ExitStack(), NoSuppress():
        return
    print("`__exit__` returning `bool | None` might suppress")
"#,
);

// A `yield` is what makes a function a generator, so one in dead code is load-bearing and
// must not be blamed, exactly as in a definitely-dead region.
testcase!(
    test_no_report_of_generator_yield_after_with,
    r#"
from typing import Iterator

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def never_yields() -> Iterator[int]:
    with NoSuppress(), NoSuppress():
        return
    yield 1

def reports_past_the_yields() -> Iterator[int]:
    with NoSuppress(), NoSuppress():
        return
    yield 1
    print("dead")  # E: This code is unreachable
"#,
);

// An `elif` test and a `case` guard are evaluated before their branch runs, exactly like the
// leading `if` test, so an exception from one is suppressible too. Spelling the same logic as
// `else: if ...` must not change the answer.
testcase!(
    test_with_exception_in_a_branch_header,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def may_raise_bool() -> bool: ...

def elif_test(flag: bool) -> None:
    while True:
        with Suppress():
            if flag:
                break
            elif may_raise_bool():
                break
            else:
                break
        print("reachable")

def match_guard(n: int) -> None:
    while True:
        with Suppress():
            match n:
                case 1 if may_raise_bool():
                    break
                case _:
                    break
        print("reachable")

def match_pattern(n: object) -> None:
    while True:
        with Suppress():
            match n:
                case [1, 2]:
                    break
                case _:
                    break
        print("reachable")
"#,
);

// Recognizing that a manager suppresses is not confined to the reachability diagnostic: the
// same predicate decides whether a function can fall off the end of a `with`.
testcase!(
    test_with_overloaded_exit_affects_implicit_return,
    r#"
from types import TracebackType
from typing import overload

class Suppressing:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> None: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> bool: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

class NotSuppressing:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> bool: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> None: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

def falls_off_the_end() -> int:  # E: missing an explicit `return`
    with Suppressing():
        return 1

def cannot_fall_off_the_end() -> int:
    with NotSuppressing():
        return 1

# Only the overload taking exception arguments decides suppression, so the code after a `with`
# on the reversed manager really is dead.
def dead_after_reversed(x: int) -> None:
    with NotSuppressing():
        raise ValueError
    print("dead")  # E: This code is unreachable
"#,
);

// Overloads are the only way to spell "suppresses, but returns `None` on the normal path".
// Suppression is decided by the call made with exception arguments, so the overload selected
// there settles it; a plain `-> bool | None` remains non-suppressing (see `NoSuppress4`).
testcase!(
    test_with_suppression_overloaded_exit,
    r#"
from types import TracebackType
from typing import assert_type, overload

class CM:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> None: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> bool: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

def f(x: int | str) -> None:
    if isinstance(x, int):
        with CM():
            raise ValueError
    assert_type(x, int | str)
"#,
);
