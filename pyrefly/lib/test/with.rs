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

def nested_ret(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                return
    assert_type(x, str)

def nested_raise(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                raise ValueError
    assert_type(x, int | str)
"#,
);

// The overload selected when an exception is in flight returns `bool`, so this context
// manager can suppress. But `context_value_exit` unions the results of calling `__exit__`
// with and without exception arguments, giving `bool | None`, which we treat as
// non-suppressing. These overloads are the only way to spell "suppresses, but returns
// `None` on the normal path": a plain `-> bool | None` is deliberately non-suppressing
// (see `NoSuppress4` above).
testcase!(
    bug = "Overloaded `__exit__` suppressing only on the exception overload is not recognized",
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
    assert_type(x, str)  # should be `int | str`
"#,
);
