/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_first_use_reads_name_twice,
    r#"
def f():
    x = ["test"]
    y = g(x, x)  # E: Argument `list[str]` is not assignable to parameter `b` with type `list[int]` in function `g`
def g(a: list[str], b: list[int]) -> None:
    pass
"#,
);

testcase!(
    test_first_use_reads_name_twice_narrowing,
    r#"
from typing import assert_type
def test(x: list[int]) -> tuple[list[int], int]:
    y = x
    return (y, 0 if 0 in assert_type(y, list[int]) else 1)
"#,
);

testcase!(
    test_empty_list_class,
    r#"
from typing import assert_type, Any
x = []
assert_type(x, list[Any])
"#,
);

testcase!(
    test_simple_int_operation_in_loop,
    r#"
from typing import assert_type, Literal
x = 5
while True:
  x = x + 1
y = x
assert_type(y, int)
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

testcase!(
    test_empty_list_append_pow,
    r#"
from typing import assert_type
def f(a: int, b: int, c: int) -> None:
    x = []
    x.append(pow(a, b, c))
    assert_type(x, list[int])
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
    assert_type(x, int | Any)
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
    test_no_infer_with_first_use_for_user_defined_generic,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import Any, assert_type
class Box[T]:
    x: T | None = None
b = Box()
b.x = 1
assert_type(b, Box[Any])
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
from typing import assert_type, Any
x = []
print(x)
x.append(1)
assert_type(x, list[Any])
"#,
);

// This used to fail when merging branches that all pointed to the same idx produced
// a Forward, because the equality check wouldn't compose across nested branches.
testcase!(
    test_inference_when_first_use_comes_after_nested_control_flow,
    r#"
from typing import assert_type, Any
x = []
if True:
    if False:
        pass
x.append(1)
assert_type(x, list[int])
"#,
);

fn env_two_exported_pins() -> TestEnv {
    TestEnv::one(
        "two_exported_pins",
        r#"
x = []
y = x.append(1)
z = x.append("1") # E: `Literal['1']` is not assignable to parameter `object` with type `int`
"#,
    )
}

testcase!(
    first_use_pins_type_when_exporting_simple_a,
    env_two_exported_pins(),
    r#"
from typing import assert_type
from two_exported_pins import y
assert_type(y, None)
from two_exported_pins import x
assert_type(x, list[int])
"#,
);

testcase!(
    first_use_pins_type_when_exporting_simple_b,
    env_two_exported_pins(),
    r#"
from typing import assert_type
from two_exported_pins import z
assert_type(z, None)
from two_exported_pins import x
assert_type(x, list[int])
"#,
);

fn env_first_use_nonpin_and_two_exported_pins() -> TestEnv {
    TestEnv::one(
        "first_use_nonpin_and_two_exported_pins",
        r#"
x = []
print(x)  # (first use does not pin type of x)
y = x.append(1)
z = x.append("1")
"#,
    )
}

testcase!(
    first_use_nonpin_and_two_exported_pins_a,
    env_first_use_nonpin_and_two_exported_pins(),
    r#"
from typing import assert_type, Any
from first_use_nonpin_and_two_exported_pins import y
assert_type(y, None)
from first_use_nonpin_and_two_exported_pins import x
assert_type(x, list[Any])
"#,
);

testcase!(
    first_use_nonpin_and_two_exported_pins_b,
    env_first_use_nonpin_and_two_exported_pins(),
    r#"
from typing import assert_type, Any
from first_use_nonpin_and_two_exported_pins import z
assert_type(z, None)
from first_use_nonpin_and_two_exported_pins import x
assert_type(x, list[Any])
"#,
);

fn env_inconsistent_pins_for_non_name_assign_placeholder() -> TestEnv {
    TestEnv::one(
        "inconsistent_pins_for_non_name_assign_placeholder",
        r#"
x, _ = [], 5
y = x.append(1)
z = x.append("1")
"#,
    )
}

testcase!(
    inconsistent_pins_for_non_name_assign_placeholder_a,
    env_inconsistent_pins_for_non_name_assign_placeholder(),
    r#"
from typing import assert_type, Any
from inconsistent_pins_for_non_name_assign_placeholder import y
assert_type(y, None)
from inconsistent_pins_for_non_name_assign_placeholder import x
assert_type(x, list[Any])
"#,
);

testcase!(
    inconsistent_pins_for_non_name_assign_placeholder_b,
    env_inconsistent_pins_for_non_name_assign_placeholder(),
    r#"
from typing import assert_type, Any
from inconsistent_pins_for_non_name_assign_placeholder import z
assert_type(z, None)
from inconsistent_pins_for_non_name_assign_placeholder import x
assert_type(x, list[Any])
"#,
);

fn env_chained_first_use_with_inconsistent_pins() -> TestEnv {
    TestEnv::one(
        "chained_first_use_with_inconsistent_pins",
        r#"
x = []
w = x
y = x.append(1)
z = w.append("1")
"#,
    )
}

testcase!(
    chained_first_use_with_inconsistent_pins_a,
    env_chained_first_use_with_inconsistent_pins(),
    r#"
from typing import assert_type, Any
from chained_first_use_with_inconsistent_pins import y
assert_type(y, None)
from chained_first_use_with_inconsistent_pins import x
assert_type(x, list[Any])
"#,
);

testcase!(
    chained_first_use_with_inconsistent_pins_b,
    env_chained_first_use_with_inconsistent_pins(),
    r#"
from typing import assert_type, Any
from chained_first_use_with_inconsistent_pins import z
assert_type(z, None)
from chained_first_use_with_inconsistent_pins import x
assert_type(x, list[Any])
"#,
);

// Tests for partial type inference in loops.
// These tests verify that first-use detection works correctly through phi nodes
// when BoundName bindings are deferred until after AST traversal.

testcase!(
    test_partial_type_first_use_in_for_loop,
    r#"
from typing import assert_type
x = []
for i in range(5):
    x.append(i)
assert_type(x, list[int])
"#,
);

testcase!(
    test_partial_type_first_use_in_while_loop,
    r#"
from typing import assert_type
x = []
i = 0
while i < 5:
    x.append(i)
    i += 1
assert_type(x, list[int])
"#,
);

testcase!(
    test_partial_type_first_use_in_nested_loops,
    r#"
from typing import assert_type
x = []
for i in range(5):
    for j in range(3):
        x.append(i + j)
assert_type(x, list[int])
"#,
);

testcase!(
    test_partial_type_secondary_read_in_loop,
    r#"
from typing import assert_type
x = []
for i in range(5):
    x.append(i)
    y = len(x)  # secondary read of x, doesn't reassign
assert_type(x, list[int])
"#,
);

testcase!(
    test_empty_container_constructor_call,
    r#"
from typing import assert_type

x = list()
x.append(1)
assert_type(x, list[int])

y = set()
y.add(2)
assert_type(y, set[int])

z = dict()
z['k'] = 3
assert_type(z, dict[str, int])
    "#,
);

testcase!(
    bug = "Container contents should be promoted",
    test_redundant_empty_container_constructor_call,
    r#"
from typing import assert_type

x = list([])
x.append(1)
assert_type(x, list[int])  # E: assert_type(list[Literal[1]], list[int])

y = dict({})
y['k'] = 3
assert_type(y, dict[str, int])  # E: assert_type(dict[Literal['k'], Literal[3]], dict[str, int])
    "#,
);

testcase!(
    test_container_of_unknown_function_parameter,
    r#"
from typing import Any, assert_type
def f(a):
    x = list(a)
    x.append(1)
    assert_type(x, list[Any])

    y = set(a)
    y.add(2)
    assert_type(y, set[Any])

    z = dict(a)
    z['k'] = 3
    assert_type(z, dict[Any, Any])
    "#,
);

testcase!(
    test_partial_quantified_in_function,
    r#"
from typing import assert_type

def f[T](x: T | None) -> list[T]: ...

x = f(None)
x.append(5)
assert_type(x, list[int])
    "#,
);

testcase!(
    test_partial_quantified_in_class_constructor,
    r#"
from typing import assert_type

class A[T]:
    def __new__[T2](cls, x: T2 | None) -> A[T2]: ...
    def add(self, x: T): ...

a = A(None)
a.add(5)
assert_type(a, A[int])
    "#,
);

testcase!(
    test_should_not_infer_on_first_use_if_solved_to_any,
    r#"
from typing import Any, assert_type

def f[T](x: list[T]) -> list[T]:
    return x

def g(x: Any):
    y = f(x)
    y.append(1)
    assert_type(y, list[Any])

# Make sure Any::Error behaves the same way as Any::Explicit
def h(x: ThisIsANameError):  # E: Could not find name
    y = f(x)
    y.append(1)
    assert_type(y, list[Any])
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/3088
testcase!(
    test_dict_with_initial_none_values,
    r#"
def parse(obj):
    data = {"a": None, "b": None, "c": None, "d": None, "e": None, "f": None, "g": None}

    for item in obj.select("div"):
        child = item.next_sibling.select_one("span")

        if item.string == "list_case":
            value = [x.strip() for x in child.strings][:-1]
        elif item.string == "update_case":
            data.update(
                {"a": child.select_one("a"), "b": child.select_one("b"), "c": None}
            )
            continue
        else:
            value = child.text

        data[item.string] = value
    return data
    "#,
);
