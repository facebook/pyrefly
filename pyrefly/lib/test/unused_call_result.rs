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

// R1/R4: discarded call with informative return type fires the rule.
testcase!(
    discarded_call_result,
    env(),
    r#"
def combine(a: list[int], b: list[int]) -> list[int]:
    return a + b

items = [1, 2, 3]
combine(items, [4, 5])  # E: is not used
"#,
);

// R1: assignment does not fire.
testcase!(
    assigned_call_result_no_error,
    env(),
    r#"
def combine(a: list[int], b: list[int]) -> list[int]:
    return a + b

items = [1, 2, 3]
x = combine(items, [4, 5])
"#,
);

// I2: call returning None does not fire.
testcase!(
    call_returning_none_no_error,
    env(),
    r#"
def side_effect() -> None:
    pass

side_effect()
"#,
);

// I2: builtin print returns None — no error.
testcase!(
    print_no_error,
    env(),
    r#"
items = [1, 2, 3]
print(items)
"#,
);

// I2: call returning Any does not fire.
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

// I2: call through Any-typed callable does not fire.
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

// I2: call returning Never/NoReturn does not fire.
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

// R1: discarded method call fires.
testcase!(
    discarded_method_call,
    env(),
    r#"
class Foo:
    def bar(self) -> int:
        return 1

Foo().bar()  # E: is not used
"#,
);

// R1: discarded constructor call fires.
testcase!(
    discarded_constructor_call,
    env(),
    r#"
class Foo:
    pass

Foo()  # E: is not used
"#,
);

// R4: bare name statement does not fire (not a call).
testcase!(
    bare_name_no_error,
    env(),
    r#"
x = 1
x
"#,
);

// R4: comparison statement does not fire (not a call).
testcase!(
    comparison_no_error,
    env(),
    r#"
x = 1
x == 1
"#,
);

// R4: attribute access statement does not fire (not a call).
testcase!(
    attribute_no_error,
    env(),
    r#"
class Foo:
    x: int = 1

Foo.x
"#,
);

// R5: discarded coroutine call fires exactly unused-coroutine, NOT unused-call-result,
// even when unused-call-result is enabled.
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

// R2: rule is off by default; no errors without enable_unused_call_result_error.
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
