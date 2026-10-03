/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_atribute_of_explicit_any_is_any,
    r#"
from typing import Any, reveal_type
def f(a: Any):
    reveal_type(a.foo)  # E: revealed type: Any
"#,
);

testcase!(
    test_subscript_of_explicit_any_is_any,
    r#"
from typing import Any, reveal_type
def f(a: Any):
    reveal_type(a[0])   # E: revealed type: Any
"#,
);

testcase!(
    test_call_on_explicit_any_is_any,
    r#"
from typing import Any, reveal_type
def f(a: Any):
    reveal_type(a())    # E: revealed type: Any
"#,
);

testcase!(
    test_binop_on_explicit_any_is_any,
    r#"
from typing import Any, reveal_type
def f(a: Any):
    reveal_type(a + 1)  # E: revealed type: Any
    reveal_type(1 + a)  # E: revealed type: Any
"#,
);
