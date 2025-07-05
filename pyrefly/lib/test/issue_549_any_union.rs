/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_any_union_attribute_access,
    r#"
from typing import Any

class A:
    pass

def foo(x: A | Any) -> None:
    x.bar  # Should not error, Any allows any attribute
"#,
);

testcase!(
    test_any_union_method_call,
    r#"
from typing import Any

class A:
    pass

def foo(x: A | Any) -> None:
    x.bar()  # Should not error, Any allows any method call
"#,
);

testcase!(
    test_non_any_union_attribute_access,
    r#"
class A:
    pass

class B:
    pass

def foo(x: A | B) -> None:
    x.bar  # E: Object of class `A` has no attribute `bar`  # E: Object of class `B` has no attribute `bar`
"#,
);
