/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_str_to_str,
    r#"
def f(x: str) -> None:
    y = str(x)  # E: Unnecessary `str()` call; argument is already of type `str`
"#,
);

testcase!(
    test_int_to_int,
    r#"
def f(x: int) -> None:
    y = int(x)  # E: Unnecessary `int()` call; argument is already of type `int`
"#,
);

testcase!(
    test_float_to_float,
    r#"
def f(x: float) -> None:
    y = float(x)  # E: Unnecessary `float()` call; argument is already of type `float`
"#,
);

testcase!(
    test_int_to_str_ok,
    r#"
def f(x: int) -> None:
    y = str(x)  # OK - converting int to str
"#,
);

testcase!(
    test_bool_to_int_ok,
    r#"
def f(x: bool) -> None:
    y = int(x)  # OK - bool is a subtype of int, types are not equal
"#,
);

testcase!(
    test_any_ok,
    r#"
from typing import Any
def f(x: Any) -> None:
    y = str(x)  # OK - argument is Any, type is unknown
"#,
);

testcase!(
    test_no_args_ok,
    r#"
def f() -> None:
    y = str()  # OK - no argument
"#,
);
