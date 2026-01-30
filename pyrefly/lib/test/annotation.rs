/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    bug = "conformance: Type quotes incorrectly referring to shadowing class member. Note that this test is likely to be removed: https://github.com/python/typing/pull/2144",
    test_class_scope_quoted_annotation_bypasses_shadowing,
    r#"
from typing import assert_type
class D:
    def int(self) -> None:
        ...
    x: "int" = 0  # E: Expected a type form
assert_type(D.x, int)  # E: assert_type(Any, int) failed
"#,
);

testcase!(
    test_quoted_type_union_operator_runtime_error,
    r#"
class ClassA:
    pass
bad1: "ClassA" | int  # E: Cannot use `|` operator with forward reference string literal and type
bad2: int | "ClassA"  # E: Cannot use `|` operator with forward reference string literal and type
"#,
);
