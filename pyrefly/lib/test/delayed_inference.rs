/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
