/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;

attrs_testcase!(
    bug = "Correctly recognize field and default decorator",
    field_default_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: dict = field()

    @a.default # E: Object of class `dict` has no attribute `default`
    def _default_a(self):
        return {}

c = C() # E: Missing argument `a` in function `C.__init__`
"#,
);

attrs_testcase!(
    bug = "Recognize validator decorator",
    field_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator # E: Object of class `int` has no attribute `validator`
    def _check_x(self, attribute, value):
        if value < 0:
            raise ValueError("x must be non-negative")
"#,
);

// A field's declared type flows to its `__init__` param, so construction args are
// type-checked: passing a `str` for an `int` field is an error.
attrs_testcase!(
    test_attrs_field_no_converter_construct_typecheck,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

C("nope")  # E: not assignable to parameter `x`
"#,
);

// With a `converter=`, the `__init__` param takes the converter's input type (`str`),
// while the stored attribute keeps the declared/output type (`int`).
attrs_testcase!(
    test_attrs_field_with_converter_still_uses_converter_input,
    r#"
from typing import assert_type, reveal_type
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(converter=to_int)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: str) -> None
c = C("5")
assert_type(c.x, int)
"#,
);

// A `factory=` field is optional in `__init__`, but its param keeps the declared
// annotation type so construction args are still type-checked.
attrs_testcase!(
    test_attrs_field_factory_param_type,
    r#"
from attrs import define, field

@define
class C:
    items: list[int] = field(factory=list)

C()              # OK: factory supplies the default
C("not a list")  # E: not assignable to parameter `items`
"#,
);

// attrs raises `ValueError` at runtime when both `default=` and `factory=` are given.
attrs_testcase!(
    test_attrs_field_default_and_factory_conflict,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// The same conflict applies to the classic `attr.ib()` specifier.
attrs_testcase!(
    test_attrs_attr_ib_default_and_factory_conflict,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(default=1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// `attr.ib` accepts `default` positionally, so a positional default plus `factory=`
// is also a conflict (`field()` is keyword-only, so this can only happen via `attr.ib`).
attrs_testcase!(
    test_attrs_attr_ib_positional_default_and_factory_conflict,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// `default=Factory(...)` is the canonical desugaring of `factory=`: only `default` is
// passed, so it must NOT be reported as a conflict.
attrs_testcase!(
    test_attrs_field_default_factory_value_ok,
    r#"
from attrs import define, field, Factory

@define
class C:
    items: list[int] = field(default=Factory(list))
"#,
);

// A non-specifier call that merely happens to use `default`/`factory` keyword names is
// not a field specifier, so no conflict is reported.
attrs_testcase!(
    test_attrs_non_field_specifier_default_factory_ok,
    r#"
from attrs import define

def helper(default: int, factory: int) -> int:
    return default

@define
class C:
    x: int = helper(default=1, factory=2)
"#,
);

// NOTHING default ⇒ field is required, and the sentinel isn't checked against the annotation.
attrs_testcase!(
    test_attrs_field_nothing_default_required,
    r#"
import attr
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING)

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Recognized through a variable, since identity follows local bindings.
attrs_testcase!(
    test_attrs_field_nothing_default_via_variable,
    r#"
import attr
from attrs import define, field

SENTINEL = attr.NOTHING

@define
class C:
    x: int = field(default=SENTINEL)

C()  # E: Missing argument `x`
"#,
);

// Also works for `attr.ib()` and a bare `from attr import NOTHING`.
attrs_testcase!(
    test_attrs_attr_ib_nothing_default_required,
    r#"
import attr
from attr import NOTHING

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(default=NOTHING)

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Counts as "no default" for ordering: can't follow a field that has one.
attrs_testcase!(
    test_attrs_field_nothing_default_ordering,
    r#"
import attr
from attrs import define, field

@define
class C:
    a: int = field(default=5)
    b: int = field(default=attr.NOTHING)  # E: without a default may not follow
"#,
);

// Mixed class: NOTHING field required, real-default field optional, declared param types.
attrs_testcase!(
    test_attrs_field_nothing_default_init_signature,
    r#"
import attr
from typing import reveal_type
from attrs import define, field

@define
class C:
    a: int = field(default=attr.NOTHING)
    b: int = field(default=5)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int, b: int = ...) -> None
"#,
);

// A real default is still type-checked: suppression must not leak to ordinary defaults.
attrs_testcase!(
    test_attrs_field_real_default_still_checked,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default="oops")  # E: `str` is not assignable to `int`
"#,
);

// `kw_only=True` is orthogonal: a NOTHING field stays required, just keyword-only.
attrs_testcase!(
    test_attrs_field_nothing_default_kw_only,
    r#"
import attr
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING, kw_only=True)

reveal_type(C.__init__)  # E: revealed type: (self: C, *, x: int) -> None
"#,
);

// A NOTHING field on a base class stays required in subclasses.
attrs_testcase!(
    test_attrs_field_nothing_default_inherited,
    r#"
import attr
from attrs import define, field

@define
class Base:
    x: int = field(default=attr.NOTHING)

@define
class Sub(Base):
    y: int = field(default=0)

Sub()   # E: Missing argument `x`
Sub(1)  # OK
"#,
);

// Regression: the NOTHING suppression must not hijack ordinary `default=` calls. The annotation
// hint must still flow into the call so an invariant generic resolves to the declared type rather
// than from the argument alone (`ContextVar[None]` vs `ContextVar[str | None]`).
attrs_testcase!(
    test_nothing_suppression_does_not_break_contextvar_default,
    r#"
from contextvars import ContextVar

x: ContextVar[str | None] = ContextVar("x", default=None)  # OK
"#,
);

// Regression: without the hint, a `default=` literal widens (`str` instead of `Literal["tcp"]`)
// and fails the invariant check.
attrs_testcase!(
    test_nothing_suppression_does_not_break_literal_default,
    r#"
from typing import Generic, Literal, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, *, default: T) -> None: ...

x: Box[Literal["tcp"]] = Box(default="tcp")  # OK
"#,
);

// A non-specifier call passing `default=attr.NOTHING` is still checked.
attrs_testcase!(
    test_attrs_non_field_call_nothing_default_still_checked,
    r#"
import attr

def f(default: object) -> str:
    return ""

x: int = f(default=attr.NOTHING)  # E: `str` is not assignable to `int`
"#,
);

// `attr.ib`'s first positional arg is `default`, so positional NOTHING ⇒ required.
attrs_testcase!(
    bug = "positional attr.ib(NOTHING) still emits a spurious `_Nothing` assignment error",
    test_attrs_field_nothing_positional_required,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(attr.NOTHING)  # E: `_Nothing` is not assignable to `int`

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Inherited `field()` fields keep their declared param type in the subclass
// `__init__`, collected base-first in MRO order.
attrs_testcase!(
    test_attrs_field_inherited_param_type,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    y: str = field(default="a")

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, x: int, y: str = ...) -> None
"#,
);
