/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_redundant_cast_literals,
    r#"
from typing import cast
x = cast(int, 5)  # E: Redundant cast: `Literal[5]` is already assignable to type `int`
y = cast(str, "hello")  # E: Redundant cast: `Literal['hello']` is already assignable to type `str`
"#,
);

testcase!(
    test_redundant_cast_variables,
    r#"
from typing import cast
x: int = 42
y = cast(int, x)  # E: Redundant cast: `int` is already assignable to type `int`
"#,
);

testcase!(
    test_valid_casts_no_warning,
    r#"
from typing import Any, cast
def deserialize(data: Any) -> int:
    return cast(int, data)  # No warning - valid cast from Any

obj: object = "hello"
s = cast(str, obj)  # No warning - valid cast from object to str
"#,
);
