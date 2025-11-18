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
