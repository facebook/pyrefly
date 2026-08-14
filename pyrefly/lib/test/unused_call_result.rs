/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn env() -> TestEnv {
    TestEnv::new().enable_unused_call_result_error()
}

testcase!(
    discarded_call_result,
    env(),
    r#"
def combine(a: list[int], b: list[int]) -> list[int]:
    return a + b

items = [1, 2, 3]
combine(items, [4, 5])  # E: is not used
x = combine(items, [4, 5]) # OK
"#,
);

testcase!(
    call_returning_none_no_error,
    env(),
    r#"
def side_effect() -> None:
    pass

side_effect()

items = [1, 2, 3]
print(items)
"#,
);

testcase!(
    call_returning_any_no_error,
    env(),
    r#"
from typing import Any

def get_any() -> Any:
    ...

get_any()
"#,
);

testcase!(
    any_typed_callable_no_error,
    env(),
    r#"
from typing import Any

def make_any() -> Any:
    ...

f: Any = make_any()
f()
"#,
);

testcase!(
    extends_any_no_error,
    env(),
    r#"
from typing import Any

class Foo(Any): ...

Foo()
"#,
);

testcase!(
    call_returning_never_no_error,
    env(),
    r#"
from typing import NoReturn

def bail() -> NoReturn:
    raise RuntimeError("bail")

bail()
"#,
);

testcase!(
    discarded_method_call,
    env(),
    r#"
class Foo:
    def bar(self) -> int:
        return 1

Foo().bar()  # E: is not used
Foo()  # E: is not used
Foo().bar  # OK, we didn't call `bar`
"#,
);

testcase!(
    discarded_coroutine_fires_unused_coroutine_not_unused_call_result,
    env(),
    r#"
async def foo() -> int:
    return 1

async def bar() -> None:
    foo()  # E: Did you forget to `await`?
"#,
);

testcase!(
    directives_no_error,
    env(),
    r#"
from typing import reveal_type, assert_type

x: int = 1
reveal_type(x)  # E: revealed type: int
assert_type(x, int)
"#,
);

// Special exports that emit their own diagnostic when used as a bare
// statement are exempt, so we don't double-report.
testcase!(
    special_export_no_unused_call_error,
    env(),
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple

TypeVar("T")  # E: TypeVar must be assigned to a variable
ParamSpec("P")  # E: ParamSpec must be assigned to a variable
TypeVarTuple("Ts")  # E: TypeVarTuple must be assigned to a variable
"#,
);

// Ordinary informative-returning builtins (e.g. `len`, `str`) are still
// flagged: only the special exports that emit their own diagnostic are exempt.
testcase!(
    informative_builtin_call_result_flagged,
    env(),
    r#"
items = [1, 2, 3]
len(items)  # E: is not used
str(items)  # E: is not used
"#,
);

testcase!(
    off_by_default,
    TestEnv::new(),
    r#"
def combine(a: list[int], b: list[int]) -> list[int]:
    return a + b

items = [1, 2, 3]
combine(items, [4, 5])
"#,
);

testcase!(
    closure,
    env(),
    r#"
from typing import Callable
def get_fn(x: int) -> Callable[[], None]:
  def foo() -> None:
    print(x)
  return foo

# should not error
get_fn(5)()
  "#,
);
