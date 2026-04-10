/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_typeform_recognized,
    r#"
from typing_extensions import TypeForm

x: TypeForm[int]
y: TypeForm[str | None]
    "#,
);

testcase!(
    test_typeform_bad_specialization,
    r#"
from typing_extensions import TypeForm

x: TypeForm[int, str]  # E: Expected 1 type argument for `TypeForm`, got 2
    "#,
);

testcase!(
    test_typeform_covariance,
    r#"
from typing_extensions import TypeForm

def get_int_form() -> TypeForm[int]:
    return int

x: TypeForm[int | str] = get_int_form()
y: TypeForm[str] = get_int_form()  # E: `TypeForm[int]` is not assignable
    "#,
);

testcase!(
    test_typeform_type_subtype,
    r#"
from typing_extensions import TypeForm

def get_type() -> type[int]:
    return int

x: TypeForm[int | str] = get_type()
y: TypeForm[str] = get_type()  # E: is not assignable
    "#,
);

testcase!(
    test_typeform_bare,
    r#"
from typing import Any, assert_type
from typing_extensions import TypeForm

def f(x: TypeForm) -> None:
    assert_type(x, TypeForm[Any])

f(int)
f(str)
    "#,
);

testcase!(
    test_typeform_assignment,
    r#"
from typing import Any, Optional
from typing_extensions import TypeForm

ok1: TypeForm[str | None] = str | None
ok2: TypeForm[str | None] = str
ok3: TypeForm[str | None] = Optional[str]
ok4: TypeForm[str | None] = None
ok5: TypeForm[Any] = int
ok6: TypeForm[str | None] = Any
ok7: TypeForm[Any] = Any

err1: TypeForm[str | None] = str | int  # E: is not assignable
    "#,
);

testcase!(
    test_typeform_reject_invalid,
    r#"
from typing import Optional, TypeVarTuple, Unpack
from typing_extensions import TypeForm

Ts = TypeVarTuple("Ts")

# Expressions that are not valid type expressions should not evaluate to a TypeForm type.
bad1: TypeForm = Unpack[Ts]  # E: `type[*TypeVarTuple[Ts]]` is not assignable to `TypeForm[Any]`
bad2: TypeForm = Optional  # E: `type[Optional]` is not assignable to `TypeForm[Any]`
    "#,
);

testcase!(
    test_typeform_callable,
    r#"
from typing import assert_type
from typing_extensions import TypeForm

x1 = TypeForm(str | None)
assert_type(x1, TypeForm[str | None])

x2 = TypeForm(int)
assert_type(x2, TypeForm[int])

x3 = TypeForm("list[int]")
assert_type(x3, TypeForm[list[int]])

x4 = TypeForm()  # E: `TypeForm` expected 1 positional argument, got 0
x5 = TypeForm(int, str)  # E: `TypeForm` expected 1 positional argument, got 2
x6 = TypeForm(type(1))  # E:
x7 = TypeForm("type(1)")  # E:
    "#,
);

testcase!(
    test_typeform_union_type,
    r#"
import types

# At runtime, str | None creates a types.UnionType object.
v: types.UnionType = str | None
    "#,
);
