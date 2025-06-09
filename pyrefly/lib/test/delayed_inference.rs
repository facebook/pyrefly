/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_empty_list_class,
    r#"
from typing import assert_type, Any
x = []
assert_type(x, list[Any])
"#,
);

testcase!(
    test_empty_list_is_generic,
    r#"
from typing import assert_type
def foo[T](x: list[T], y: list[T]) -> T: ...
r = foo([], [1])
assert_type(r, int)
"#,
);

testcase!(
    test_empty_list_append,
    r#"
from typing import assert_type
x = []
x.append(4)
assert_type(x, list[int])
"#,
);

testcase!(
    test_empty_list_check,
    r#"
from typing import Literal, assert_type
x = []
def f(x: list[Literal[4]]): ...
f(x)
assert_type(x, list[Literal[4]])
"#,
);

// NOTE(grievejia): There's also an argument to be made that `y` should be inferred as
// `list[int] | list[Any]`, and `e` inferred as `int | Any`. The test case here is to ensure
// that if we ever want to take the alternative behavior, an explicit acknowledgement (of
// changing this test case) is required.
testcase!(
    test_or_empty_list,
    r#"
from typing import assert_type
def test(x: list[int]) -> None:
    y = x or []
    assert_type(y, list[int])
    for e in y:
        assert_type(e, int)
"#,
);

testcase!(
    test_solver_variables,
    r#"
from typing import assert_type, Any

def foo[T](x: list[T]) -> T: ...

def bar():
    if False:
        return foo([])
    return foo([])

assert_type(bar(), Any)
"#,
);

testcase!(
    test_solver_variables_2,
    r#"
from typing import assert_type, Any
def foo[T](x: list[T]) -> T: ...
def bar(random: bool):
    if random:
        x = foo([])
    else:
        x = foo([1])
    assert_type(x, int)
    "#,
);

testcase!(
    test_deferred_type_for_user_defined_generic,
    r#"
from typing import assert_type
class Box[T]:
    x: T | None = None
b = Box()
b.x = 1
assert_type(b, Box[int])
    "#,
);

testcase!(
    test_deferred_type_for_indeterminate_generic_function_output,
    r#"
from typing import assert_type
def new_empty_list[T]() -> list[T]:
    ...
x = new_empty_list()
x.append(1)
assert_type(x, list[int])
"#,
);

testcase!(
    test_inference_when_first_use_does_not_determine_type,
    r#"
from typing import assert_type
x = []
print(x)
x.append(1)
assert_type(x, list[int])
"#,
);

testcase!(
    test_inference_when_first_use_comes_after_nested_control_flow,
    r#"
from typing import assert_type
x = []
if True:
    if False:
        pass
x.append(1)
assert_type(x, list[int])
"#,
);

fn env_nondeterminism() -> TestEnv {
    TestEnv::one(
        "nondeterministic",
        r#"
x = []
y = x.append(1)
z = x.append("1")
"#,
    )
}

testcase!(
    bug = "Forcing y first causes x to be list[int]. See version b for why this is a bug.",
    emulate_nondeterminism_a,
    env_nondeterminism(),
    r#"
from typing import assert_type
from nondeterministic import y
assert_type(y, None)
from nondeterministic import x
assert_type(x, list[int])
"#,
);

/*
// This test occasionally fails, particularly under cargo - it usually behaves as the assert_type
// indicates, but it appears that in some cases the analysis of `nondeterminism` and `main` can race
// one another, and we actually get nondeterministic test results.
testcase!(
    bug = "Forcing z first causes x to be list[str]. See version a for why this is a bug.",
    emulate_nondeterminism_b,
    env_nondeterminism(),
    r#"
from typing import assert_type
from nondeterministic import z
assert_type(z, None)
from nondeterministic import x
assert_type(x, list[str])
"#,
);
*/
