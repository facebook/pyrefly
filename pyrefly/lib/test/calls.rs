/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_generic_call_happy_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: T) -> tuple[S, T]: ...
force_error(f(1, "foo"))  # E: Argument `tuple[int, str]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_simple,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S) -> tuple[S, T]: ...
force_error(f(1))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_union_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: list[T] | None) -> tuple[S, T]: ...
force_error(f(1, None))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_self_type_subst,
    r#"
from typing import assert_type, Self
class A:
    def __new__(cls) -> Self: ...
class B[T](A): ...
class C[T]: ...
assert_type(A.__new__(A), A)
assert_type(A.__new__(B[int]), B[int])
assert_type(A.__new__(C[int]), Self)
    "#,
);

testcase!(
    test_deprecated_call,
    r#"
from warnings import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...
old_function()  # E: Call to deprecated function `old_function`
    "#,
);
