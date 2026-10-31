/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;
use crate::testcase;

testcase!(
    test_unwrap_recursive_annotation,
    r#"
from inspect import unwrap
from typing import Any, assert_type

class Wrapper[T]:
    __wrapped__: "Wrapper[list[T]]"
    def __call__(self) -> int:
        return 1

def check(value: Wrapper[int]) -> None:
    assert_type(unwrap(value), Any)
    "#,
);

testcase!(
    test_unwrap_argument_errors_after_mutation,
    r#"
from functools import update_wrapper
from inspect import unwrap
from typing import Any, assert_type

def source() -> str:
    return "source"

def target() -> bool:
    return False

update_wrapper(target, source)
assert_type(unwrap(42), Any)
unwrap(missing)  # E: Could not find name `missing`
    "#,
);

testcase!(
    test_shadowed_update_wrapper_preserves_unwrap,
    r#"
from functools import wraps
from inspect import unwrap
from typing import assert_type

def source() -> str:
    return "source"

@wraps(source)
def wrapped() -> int:
    return 1

def update_wrapper(target: object, source: object) -> None:
    pass

update_wrapper(wrapped, source)
assert_type(unwrap(wrapped)(), str)
    "#,
);

#[test]
fn test_unwrap_depth_limit_no_panic() {
    for limit in [8, 16, 32] {
        // Depth-limit diagnostics are allowed; this smoke test requires safe placeholder handling.
        let _ = testcase_for_macro(
            TestEnv::new().with_recursion_depth_limit(limit),
            "from inspect import unwrap\nunwrap(42)\n",
            file!(),
            line!(),
        );
    }
}
