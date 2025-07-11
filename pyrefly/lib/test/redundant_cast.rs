/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

 use crate::testcase;


testcase!(
    test_redundant_cast_simple,
    r#"
from typing import cast

x = cast(int, 5)  # E: Redundant cast: expression already has type `int`
y = cast(str, "hello")  # E: Redundant cast: expression already has type `str`
"#,
);

testcase!(
    test_redundant_cast_with_annotation,
    r#"
from typing import cast

x: int = cast(int, 42)  # E: Redundant cast: expression already has type `int`
"#,
);

testcase!(
    test_non_redundant_cast,
    r#"
from typing import Any, cast

def deserialize(data: Any) -> int:
    return cast(int, data)
"#,
);
