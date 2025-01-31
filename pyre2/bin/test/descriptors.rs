/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;
use crate::testcase_with_bug;

testcase!(
    test_staticmethod_with_explicit_parameter_type,
    r#"
from typing import assert_type, reveal_type, Callable
class C:
    @staticmethod
    def foo() -> int:
        return 42
    @staticmethod
    def bar(x: int) -> int:
        return x
def f(c: C):
    assert_type(C.foo, Callable[[], int])
    assert_type(c.foo, Callable[[], int])
    reveal_type(C.bar)  # E: (x: int) -> int
    reveal_type(c.bar)  # E: (x: int) -> int
    assert_type(C.foo(), int)
    assert_type(c.foo(), int)
    assert_type(C.bar(42), int)
    assert_type(c.bar(42), int)
    "#,
);

testcase_with_bug!(
    "We do not yet correctly bind the type of the first parameter of a staticmethod",
    test_staticmethod_calls_with_implicit_parameter_type,
    r#"
from typing import assert_type, Callable, Any
class C:
    @staticmethod
    def bar(x) -> int:
        return 42
def f(c: C):
    assert_type(c.bar(42), int) # E: EXPECTED Literal[42] <: C
    assert_type(c.bar(42), int) # E: EXPECTED Literal[42] <: C
    "#,
);

testcase!(
    test_classmethod_access,
    r#"
from typing import reveal_type
class C:
    @classmethod
    def foo(cls) -> int:
        return 42
def f(c: C):
    reveal_type(C.foo)  # E: revealed type: BoundMethod[type[C], (cls: C) -> int]
    reveal_type(c.foo)  # E: revealed type: BoundMethod[type[C], (cls: C) -> int]
    "#,
);

testcase!(
    test_classmethod_calls_with_explicit_parameter_type,
    r#"
from typing import assert_type
class C:
    @classmethod
    def foo(cls: type[C]) -> int:
        return 42
def f(c: C):
    assert_type(C.foo(), int)
    assert_type(c.foo(), int)
    "#,
);

testcase_with_bug!(
    "We do not yet correctly bind the type of the first parameter of a classmethod",
    test_classmethod_calls_with_implicit_parameter_type,
    r#"
from typing import assert_type
class C:
    @classmethod
    def foo(cls) -> int:
        return 42
def f(c: C):
    assert_type(C.foo(), int)  # E: EXPECTED type[C] <: C
    assert_type(c.foo(), int)  # E: EXPECTED type[C] <: C
    "#,
);

testcase!(
    test_read_only_property,
    r#"
from typing import assert_type, reveal_type
class C:
    @property
    def foo(self) -> int:
        return 42
def f(c: C):
    assert_type(c.foo, int)
    c.foo = 42  # E: Attribute `foo` of class `C` is a read-only property and cannot be set
    reveal_type(C.foo)  # E: revealed type: property[(self: C) -> int]
    "#,
);

testcase_with_bug!(
    "Property setters are not yet supported",
    test_property_with_setter,
    r#"
from typing import assert_type, reveal_type
class C:
    @property
    def foo(self) -> int:
        return 42
    # TODO(samgoldman) This is a workaround to avoid function def folding in
    # bindings; once we are dealing with overloads in some other way we can
    # remove it.
    def _break_function_def_folding_in_bindings_(self) -> None:
        ...
    @foo.setter  # E: TODO: Expr::attr_infer attribute base undefined for type: property[(self: C) -> int] (trying to access setter)
    def foo(self, value: str) -> None:
        pass
def f(c: C):
    assert_type(c.foo, int)  # E: assert_type(Any | property[(self: C) -> int], int)
    c.foo = "42"  # (no error here because it is becoming Any)
    reveal_type(C.foo)  # E: revealed type: Error | property[(self: C) -> int]
    "#,
);
