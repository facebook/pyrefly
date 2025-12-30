/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

// LateFutureImport: `from __future__ import` statements must appear at the beginning of the file,
// after the module docstring and any comments, but before any other code.
testcase!(
    test_late_future_import,
    r#"
import os
from __future__ import annotations  # E: __future__ imports must be at the top of the file
"#,
);

testcase!(
    test_late_future_import_after_assignment,
    r#"
x = 1
from __future__ import annotations  # E: __future__ imports must be at the top of the file
"#,
);

testcase!(
    test_valid_future_import,
    r#"
from __future__ import annotations
import os
x = 1
"#,
);

testcase!(
    test_valid_future_import_after_docstring,
    r#"
"""Module docstring."""
from __future__ import annotations
import os
"#,
);

testcase!(
    test_duplicate_parameter,
    r#"
def foo(x: int, x: str):  # E: Duplicate parameter "x"
    pass
"#,
);

testcase!(
    test_rebound_comprehension_variable,
    r#"
[y := 1 for y in range(10)]  # E: assignment expression cannot rebind comprehension variable
"#,
);

testcase!(
    test_nonlocal_at_module_level,
    r#"
nonlocal x  # E: nonlocal declaration not allowed at module level
"#,
);

testcase!(
    test_nonlocal_in_class_body,
    r#"
x = 1
class Foo:
    nonlocal x  # E: Found `x`, but it is coming from the global scope
"#,
);

testcase!(
    test_nonlocal_inside_function_ok,
    r#"
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
"#,
);

testcase!(
    test_multiple_case_assignment,
    r#"
x = [1, 2]
match x:
    case [a, a]:  # E: multiple assignments to name `a` in pattern
        pass
"#,
);

testcase!(
    test_duplicate_match_key,
    r#"
x = {"a": 1, "b": 2}
match x:
    case {"a": 1, "a": 2}:  # E: mapping pattern checks duplicate key `"a"`
        pass
"#,
);

testcase!(
    test_duplicate_match_class_attribute,
    r#"
class Point:
    x: int
    y: int

p = Point()
match p:
    case Point(x=1, x=2):  # E: attribute name `x` repeated in class pattern
        pass
"#,
);

testcase!(
    test_duplicate_type_parameter_function,
    r#"
def f[T, T](x: T) -> T:  # E: duplicate type parameter
    return x
"#,
);

testcase!(
    test_duplicate_type_parameter_class,
    r#"
class C[T, T]:  # E: duplicate type parameter
    pass
"#,
);

testcase!(
    test_duplicate_type_parameter_type_alias,
    r#"
type Alias[T, T] = list[T]  # E: duplicate type parameter
"#,
);

testcase!(
    bug = "we should not error on the global declaration. Also, the error on the nonlocal declaration should be more specific.",
    test_nonlocal_and_global,
    r#"
def bar():
    nonlocal x # E: Could not find name `x`
    global x # E: `x` was assigned in the current scope before the global declaration
"#,
);

testcase!(
    test_import_star_in_function,
    r#"
def foo():
    from os import *  # E: `from os import *` only allowed at module level
"#,
);

testcase!(
    test_import_star_in_class,
    r#"
class Foo:
    from os import *  # E: `from os import *` only allowed at module level
"#,
);

testcase!(
    bug = "TODO: raise error on this invalid program",
    test_write_to_debug,
    r#"
__debug__ = False
"#,
);

testcase!(
    test_invalid_expression_in_match,
    r#"
x = 1
match x:
    case 1 + 1:  # E: Parse error: Expected an imaginary number in complex literal pattern
        pass
"#,
);

testcase!(
    test_future_feature_not_defined,
    r#"
from __future__ import not_a_real_feature  # E: Could not import `not_a_real_feature` from `__future__`
"#,
);

testcase!(
    test_async_comprehension_in_sync_comprehension,
    r#"
async def async_gen():
    for i in range(5):
        yield i

def sync_func():
    result = [x async for x in async_gen()]  # E: `async` can only be used inside an async function
    return result
"#,
);

testcase!(
    test_invalid_star_expression,
    r#"
x = *[1, 2, 3]  # E: Expected a type form, got instance of `Literal[1]` # E: Expected a type form, got instance of `Literal[2]` # E: Expected a type form, got instance of `Literal[3]`
"#,
);

testcase!(
    bug = "Raise error on this invalid program",
    test_multiple_starred_expressions,
    r#"
x, *y, *z = [1, 2, 3, 4]
"#,
);
