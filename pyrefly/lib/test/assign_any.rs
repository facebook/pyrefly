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
    assert_type(x, int | Any)
"#,
);

testcase!(
    test_assign_to_different_any,
    r#"
from typing import Any, reveal_type

def explicit_any() -> Any: ...
def implicit_any(x): return x

def explicit_to_implicit(x: Any):
    x = implicit_any(x)
    reveal_type(x)  # E: Unknown

def error_to_explicit(x: Oops):  # E:
    x = explicit_any()
    reveal_type(x)  # E: Any
    "#,
);

testcase!(
    test_branch_between_declare_and_assign,
    r#"
from typing import Any, assert_type
def get_any() -> Any: ...
def f(x: int, cond: bool):
    if cond:
        pass
    x = get_any()
    assert_type(x, int)
    "#,
);

testcase!(
    test_reassign_between_declare_and_assign_any,
    r#"
from typing import Any, assert_type
def get_any() -> Any: ...
def f(x: float):
    x = 0
    x = get_any()
    assert_type(x, Any)
    "#,
);

testcase!(
    test_unconditional_narrow_between_declare_and_assign,
    r#"
from typing import Any, assert_type
def get_any() -> Any: ...
def f(x: float):
    assert isinstance(x, int)
    x = get_any()
    assert_type(x, Any)
    "#,
);

testcase!(
    test_narrow_optional_any_to_any,
    r#"
from typing import Any, Optional, assert_type

def get_any() -> Any: ...

def f(x: Any | None):
    if x is None:
        x = get_any()
        assert_type(x, Any)
    assert_type(x, Any)

def g(x: Optional[Any]):
    if x is None:
        x = get_any()
        assert_type(x, Any)
    assert_type(x, Any)
    "#,
);

testcase!(
    test_assign_to_any_twice,
    r#"
from typing import Any, assert_type

def get_any() -> Any: ...

def f(x: int):
    x = get_any()
    x = get_any()
    assert_type(x, int)
    "#,
);

testcase!(
    test_narrow_any_union_to_any,
    r#"
from typing import Any, assert_type
def get_any() -> Any: ...
def f(x: Any | None = None):
    x = x or get_any()
    assert_type(x, Any)
    "#,
);
