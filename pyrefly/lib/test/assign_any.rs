/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_annotated_var_preserves_type_after_any_assign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: int
x = f()
assert_type(x, int)
"#,
);

testcase!(
    test_reassigned_var_preserves_annotation_over_any,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: str = "hello"
x = f()
assert_type(x, str)
"#,
);

testcase!(
    test_annotated_var_augassign_any,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: int = 0
x += f()
assert_type(x, int)
"#,
);

testcase!(
    test_annotated_var_context_manager_any,
    r#"
from typing import Any, assert_type

class CM:
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> None: ...

x: int
with CM() as x:
    assert_type(x, int)
"#,
);

testcase!(
    test_annotated_var_for_loop_any,
    r#"
from typing import Any, assert_type

xs: list[Any] = []

y: int
for y in xs:
    assert_type(y, int)
"#,
);

testcase!(
    test_nullable_annotation_any_assign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: int | None = None
x = f()
assert_type(x, int | None)
"#,
);

testcase!(
    test_param_nullable_annotation_any_reassign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

def test(x: int | None) -> None:
    x = f()
    assert_type(x, int | None)
"#,
);

testcase!(
    test_any_expr_preserves_full_union_annotation,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: int | str | None = None
x = f()
assert_type(x, int | str | None)
"#,
);

testcase!(
    test_param_concrete_annotation_any_reassign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

def test(x: int) -> None:
    x = f()
    assert_type(x, int)
"#,
);

testcase!(
    test_union_annotation_any_assign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: int | str = 0
x = f()
assert_type(x, int | str)
"#,
);

testcase!(
    test_generic_annotation_any_assign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

x: list[int] = [1, 2, 3]
x = f()
assert_type(x, list[int])
"#,
);

testcase!(
    test_param_none_guard_any_reassign,
    r#"
from typing import Any, assert_type

def f() -> Any: ...

def test(x: int | None) -> None:
    if x is None:
        x = f()
    assert_type(x, int | None)
"#,
);
