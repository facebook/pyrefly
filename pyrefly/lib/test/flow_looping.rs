/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    bug = "The results include over-eager pinning of vars in generic solving, see https://github.com/facebook/pyrefly/issues/105",
    test_loop_with_generic_pin,
    r#"
def condition() -> bool: ...
def f[T](x: T, y: list[T]) -> T: ...
x = 5
y: list[str] = []
while condition():
    x = f(x, y)  # E: Argument `list[str]` is not assignable to parameter `y` with type `list[int]` in function `f`
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/683
testcase!(
    test_loop_with_sized_in_inner_iteration,
    r#"
def f(xs: list[list]):
    for x in xs:
        for i in range(len(x)):
            x[i] = 1
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/812
testcase!(
    test_loop_with_set_and_len,
    r#"
def f(my_set: set[int]):
    while True:
        start_size = len(my_set)
        my_set.update([])
        if len(my_set) == start_size:
            return
"#,
);

// Regression test: at one point, excessive loop recursion caused the reveal type to be `Unknown`
testcase!(
    test_loop_with_dict_get,
    r#"
from typing import reveal_type
def f(keys: list[str]):
    counters: dict[str, int] = {}
    for k in keys:
        counters[k] = reveal_type(counters.get(k, 0))  # E: revealed type: int
"#,
);

testcase!(
    test_while_simple,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = None
    while condition():
        assert_type(x, Literal["hello world"] | None)
        x = "hello world"
        assert_type(x, Literal["hello world"])
    assert_type(x, Literal["hello world"] | None)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1701
testcase!(
    test_while_condition_is_in_loop,
    r#"
def main():
    q = 1
    while q:
        q -= 1
    "#,
);

testcase!(
    bug = "A recursive redefinition in a loop produces a hard-to-follow error message + location",
    test_while_creates_recursive_type,
    r#"
from typing import assert_type, Any, Literal
def f(condition) -> None:
    x = 1
    # It's fine to error here, but ideally we would error at the assignment
    # rather than the `while`, and ideally we would note that the type is recursive
    # in a way we don't support.
    while condition():  # E: `Literal[1] | list[int]` is not assignable to `int`
        assert_type(x, Literal[1] | list[int])
        x = [x]
        assert_type(x, list[int])
    assert_type(x, Literal[1] | list[int])
    "#,
);

testcase!(
    test_while_noop,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = 1
    while condition():
        pass
    assert_type(x, Literal[1])
    "#,
);

testcase!(
    test_while_fancy_noop,
    r#"
from typing import assert_type, Any, Literal
def f(condition) -> None:
    x = 1
    while condition():
        x = x
    assert_type(x, Literal[1])
    "#,
);

testcase!(
    test_while_if,
    r#"
from typing import assert_type, Any, Literal
def f(condition1, condition2) -> None:
    x = None
    while condition1():
        if condition2():
            x = "hello"
    assert_type(x, Literal['hello'] | None)
    "#,
);

testcase!(
    test_while_two_vars,
    r#"
from typing import assert_type, Any, Literal
def f(cond1, cond2, cond3) -> None:
    x = 1
    y = ""
    while cond1():
        if cond2():
            x = y
        if cond3():
            y = x
    assert_type(x, Literal["", 1])
    assert_type(y, Literal["", 1])
    "#,
);

testcase!(
    test_while_else,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = None
    while condition():
        x = 1
    else:
        x = ""
    assert_type(x, Literal[""])
    "#,
);

testcase!(
    test_while_break_else,
    r#"
from typing import assert_type, Any, Literal
def f(cond1, cond2) -> None:
    x = None
    while cond1():
        if cond2():
            x = "value"
            break
        else:
            x = "overwritten"
    else:
        assert_type(x, Literal["overwritten"] | None)
        x = "default"
    assert_type(x, Literal["default", "value"])
    "#,
);

testcase!(
    test_while_else_while,
    r#"
while False:
    x = 0
else:
    while False:
        x = 1
    "#,
);

testcase!(
    test_while_infinite_implicit_return,
    r#"
def f1(b) -> int:
    while True:
        if b():
            return 1

def f2(b) -> int:  # E: Function declared to return `int` but is missing an explicit `return`
    while True:
        if b():
            break
    "#,
);

testcase!(
    test_while_reassignment_with_annotation,
    r#"
from typing import assert_type, Literal
def f(cond):
    x: int = 0
    while cond():
        x: int = 1
    assert_type(x, int)
    "#,
);

testcase!(
    test_for_simple,
    r#"
from typing import assert_type
def f(x: list[int]) -> None:
    for i in x:
        assert_type(i, int)
    assert_type(i, int)
    "#,
);

testcase!(
    test_for_tuple,
    r#"
from typing import assert_type
def f(x: tuple[int, str]) -> None:
    for i in x:
        assert_type(i, int | str)
    "#,
);

testcase!(
    test_for_literal_string,
    r#"
from typing import assert_type, LiteralString
for i in "abcd":
    assert_type(i, LiteralString)
    "#,
);

testcase!(
    test_for_any,
    r#"
from typing import Any, assert_type
def f(x: Any):
    for i in x:
        assert_type(i, Any)
    "#,
);

testcase!(
    test_for_reassign,
    r#"
from typing import assert_type
def f(x: list[int]):
    y = None
    for i in x:
        y = i
    assert_type(y, int | None)
    "#,
);

testcase!(
    test_for_else_reassign,
    r#"
from typing import assert_type, Literal
def f(x: list[int]):
    y = None
    for i in x:
        y = i
    else:
        y = 'done'
    assert_type(y, Literal['done'])
    "#,
);

testcase!(
    test_for_multiple_targets,
    r#"
from typing import assert_type
def f(x: list[tuple[int, str]]) -> None:
    for (i, j) in x:
        assert_type(i, int)
        assert_type(j, str)
    "#,
);

testcase!(
    test_for_scope,
    r#"
from typing import assert_type
def f(x: list[int]) -> None:
    for i in x:
        pass
    assert_type(i, int)
    "#,
);

testcase!(
    test_for_target_annot_compatible,
    r#"
def f(x: list[int]) -> None:
    i: int = 0
    for i in x:
        pass
    "#,
);

testcase!(
    test_for_target_annot_incompatible,
    r#"
def f(x: list[int]) -> None:
    i: str = ""
    for i in x: # E: Cannot use variable `i` with type `str` to iterate over elements of type `int`
        pass
    "#,
);

testcase!(
    test_for_implicit_return,
    r#"
def test1(match: float) -> float:
    for i in range(10):
        if i == match:
            return 3.14
    else:
        msg = "No value found"
        raise ValueError(msg)

def test2(match: float) -> float:  # E: Function declared to return `float` but is missing an explicit `return`
    for i in range(10):
        if i == match:
            break
    else:
        msg = "No value found"
        raise ValueError(msg)
    "#,
);

testcase!(
    test_for_with_reassign,
    r#"
from typing import assert_type, Literal
for i in range((y := 10)):
    assert_type(i, int)
    assert_type(y, Literal[10] | str)
    i = str()  # This doesn't actually flow back into the next iteration
    y = str()  # But this does
"#,
);

fn loop_export_env() -> TestEnv {
    TestEnv::one(
        "imported",
        r#"
exported = None

for _ in []:
    ignored = 1
"#,
    )
}

testcase!(
    test_loop_export,
    loop_export_env(),
    r#"
import imported
from typing import assert_type

assert_type(imported.exported, None)
"#,
);

testcase!(
    test_loop_increment,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while cond:
        n += 1
    assert_type(n, int)
"#,
);

testcase!(
    test_loop_test_and_increment,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while n < 10:
        n += 1
    assert_type(n, int)
"#,
);

testcase!(
    test_nested_loop_increment,
    r#"
from typing import assert_type, Literal
def f_toplevel(cond: bool):
    n = "n"
    if cond:
        n = 1
    else:
        n = 1.5
    while cond:
        n += 1
    assert_type(n, float | int)
while True:
    # Make sure we treat a function nested in a loop the same
    # way (i.e. that the loop in a parent scope doesn't affect
    # flow merging in function scope).
    def f_in_loop(cond: bool):
        n = "n"
        if cond:
            n = 1
        else:
            n = 1.5
        while cond:
            n += 1
        assert_type(n, float | int)
"#,
);

testcase!(
    test_loop_test_and_increment_return,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while cond:
        n += 1
    return n

assert_type(f(True), int)
"#,
);

testcase!(
    test_nested_loops_simple,
    r#"
def f(cond1: bool, cond2: bool):
    n = 0
    while cond1:
        while cond2:
            n += 1
"#,
);

testcase!(
    test_nested_loops_return,
    r#"
from typing import assert_type, Literal

def f(cond1: bool, cond2: bool):
    n = 0
    while cond1:
        while cond2:
            n += 1
    return n

assert_type(f(True, True), int)
"#,
);

testcase!(
    test_augassign_in_loop_simple,
    r#"
def f(args, cond):
    n = 0
    for arg in args:
        if cond:
            n += 1
"#,
);

testcase!(
    test_augassign_in_loop_return,
    r#"
from typing import assert_type, Literal

def f(args, cond):
    n = 0
    for arg in args:
        if cond:
            n += 1
    return n

assert_type(f([1, 2, 3], True), int)
"#,
);

testcase!(
    test_loops_and_ifs_galore,
    r#"
from typing import assert_type, Literal

def f(cond1: bool, cond2: bool, cond3: bool, cond4: bool):
    i = 0
    while cond1:
        if cond2:
            if cond3:
                pass
            if cond4:
                i += 1
    return i

assert_type(f(True, True, True, True), int)
"#,
);

testcase!(
    test_loop_defaulting,
    r#"
# From https://github.com/facebook/pyrefly/issues/104
from typing import assert_type
class Foo:
    pass

def rebase(parent: Foo | int) -> Foo: ...

def test(b: bool, x: Foo) -> None:
    while b:
        x = rebase(x)
    assert_type(x, Foo)
"#,
);

testcase!(
    test_loop_enumerate,
    r#"
# From https://github.com/facebook/pyrefly/issues/267
def foo() -> list[int]:
    results: list[int] = [1, 2, 3]
    for i, x in enumerate(results):
        results[i] = x * 10
    return results
"#,
);

testcase!(
    test_loop_nested_binding,
    r#"
# This used to fail, thinking the type was Never
def f():
    class X:
        pass

    while True:
        z = "" if True else ""
        break
    else:
        exit(1)

    x: X
"#,
);

testcase!(
    bug = "Single-shot analysis cannot handle this - see https://github.com/facebook/pyrefly/issues/1234",
    test_assign_result_of_call_back_to_argument,
    r#"
class Cursor:
    def finished(self) -> bool:
        ...

class Query:
    def send(self, cursor: Cursor | None) -> Cursor:
        ...

def test(q: Query) -> None:
    cursor = None
    while not cursor or not cursor.finished():  # E: `Cursor | None` is not assignable to `None` (caused by inconsistent types when breaking cycles)
        cursor = q.send(cursor)
"#,
);

testcase!(
    test_reveal_type_in_loop,
    r#"
# This used to get confused by what reveal_type is
from typing import *
x = 1
while True:
    reveal_type(x) # E: revealed type: Literal[1]
    break
else:
    exit(1)
"#,
);

// Test for https://github.com/facebook/pyrefly/issues/726
testcase!(
    test_reassign_literal_str_to_str_in_loop,
    r#"
import os

path = '/'
for x in ['home', 'other']:
    path = os.path.join(path, x)
    "#,
);

// Test for https://github.com/facebook/pyrefly/issues/747
testcase!(
    test_benign_reassign_and_narrow_in_loop,
    r#"
from typing import assert_type

def test(x: int | None, i: int):
    for _ in []:
        x = x or i
        assert_type(x, int)
"#,
);

testcase!(
    test_expand_loop_recursive_and_match_generic,
    r#"
from typing import assert_type
def f[T](x: list[T]) -> T: ...
def condition() -> bool: ...

good = [1]
while condition():
    good = [f(good)]
assert_type(good, list[int])

bad = [1]
while condition():  # E: `list[int] | list[str]` is not assignable to `list[int]` (caused by inconsistent types when breaking cycles)
    if condition():
        bad = [f(bad)]  # E:  Argument `list[int] | list[str]` is not assignable to parameter `x` with type `list[int]` in function `f`
    else:
        bad = [""]
"#,
);

// Test for https://github.com/facebook/pyrefly/issues/1505
testcase!(
    test_dict_get_self_assignment,
    r#"
d: dict[str, str] = {}
a: str | None = None
for i in range(10):
    a = d.get('x', a)
    "#,
);

// Test for https://github.com/facebook/pyrefly/issues/1453
testcase!(
    test_against_regression_on_1453,
    r#"
from math import gcd
from typing import assert_type
def remove_common(x: int, g: int) -> int:
    while g > 1:
        assert_type(g, int)
        assert_type(x, int)
        x //= g
        g = gcd(g, x)
        assert_type(g, int)
    return x
    "#,
);

// Test for https://github.com/facebook/pyrefly/issues/1453
testcase!(
    test_against_regression_on_1454,
    r#"
d: dict[str, str] = {}
a: str | None = None
for i in range(10):
    a = d.get('x', a)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1565
testcase!(
    test_nested_while_int_assignment,
    r#"
def decode(s: str):
    i = 0
    while i < len(s):
        j = i
        while (
            s[j] != ""
        ):
            j += 1
        i = j + 1
    "#,
);

testcase!(
    test_break_continue_outside_of_loop,
    r#"
def test():
    break # E: `break` outside loop
    continue # E: `continue` outside loop
    "#,
);

testcase!(
    test_for_range_nonzero_literal_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in range(3):
        x = "a"
    assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_for_range_one_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in range(1):
        x = "a"
    assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_for_nonempty_list_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in [1, 2, 3]:
        x = "a"
    assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_for_nonempty_tuple_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in (1, 2, 3):
        x = "a"
    assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_for_nonempty_set_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in {1, 2, 3}:
        x = "a"
    assert_type(x, Literal["a"])
    "#,
);

// These should still produce errors because the loop may not execute
testcase!(
    test_for_range_zero_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in range(0):
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_for_empty_list_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo():
    for _ in []:
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_for_dynamic_range_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo(n: int):
    for _ in range(n):
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_for_dynamic_list_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo(xs: list[int]):
    for _ in xs:
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_for_nonempty_with_break_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo(cond: bool):
    for _ in range(3):
        if cond:
            break
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_while_true_with_break_may_not_define_variable,
    r#"
from typing import assert_type, Literal
def foo(cond: bool):
    while True:
        if cond:
            break
        x = "a"
    assert_type(x, Literal["a"])  # E: `x` is uninitialized
    "#,
);

testcase!(
    test_for_nonempty_narrow_unbound_variable,
    r#"
from typing import assert_type
def foo(cond: bool):
    for _ in range(3):
        if cond:
            if x is not None:  # E: `x` is uninitialized
                pass
    print(x)  # (Note there's no error here, but that's probably ok since we error above)
    x: int | None = None
    assert_type(x, int | None)
    "#,
);
