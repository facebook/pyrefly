/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
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
    test_union_operator_with_bare_string_literal,
    r#"
from typing import assert_type, TypeVar, Generic
T = TypeVar("T")
class C(Generic[T]): ...
bad1: int | "str" = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
bad2: int | "str" | T = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
bad3: "str" | int = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
bad4: "str" | int | T = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
ok1: T | "str" = "foo"
ok2: "str" | T = "foo"
ok3 = list["str" | T]
ok4 = (int) | (str)
ok5: "str" | C[int] = "foo"
ok6: C[int] | "str" = "foo"
"#,
);

// Test that the error is raised for Python 3.13 (explicit version)
testcase!(
    test_union_operator_with_bare_string_literal_py313,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
bad1: int | "str" = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
bad2: "str" | int = "foo"  # E: Cannot use `|` operator with forward reference string literal and type
"#,
);

// Test that the error is NOT raised for Python 3.14+
// In Python 3.14+, annotations are not evaluated at runtime by default (PEP 649)
testcase!(
    test_union_type_with_bare_string_literal_py314,
    TestEnv::new_with_version(PythonVersion::new(3, 14, 0)),
    r#"
ok1: int | "str" = "foo"
ok2: "str" | int = "foo"
"#,
);

// Test that the error is NOT raised when `from __future__ import annotations` is used
// With future annotations, annotations are not evaluated at runtime
testcase!(
    test_union_type_with_bare_string_literal_future_annotations,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
from __future__ import annotations
ok1: int | "str" = "foo"
ok2: "str" | int = "foo"
"#,
);
