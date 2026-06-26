/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;
use crate::test::attrs::util::attrs_env;
use crate::testcase;

// `@a.default` supplies the default, so `a` is optional and the `a.default` access resolves.
attrs_testcase!(
    field_default_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: dict = field()

    @a.default
    def _default_a(self):
        return {}

C()    # OK
C({})  # OK
"#,
);

attrs_testcase!(
    field_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check_x(self, attribute, value):
        if value < 0:
            raise ValueError("x must be non-negative")

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// `in_(EnumClass)` infers `_ValidatorType[object]`, but the annotation stays authoritative (#3429).
attrs_testcase!(
    test_attrs_field_validator_does_not_widen_annotation,
    r#"
from enum import Enum
from attrs import define, field, validators

class Color(Enum):
    RED = 1
    GREEN = 2

@define
class C:
    color: Color = field(validator=validators.in_(Color))

C(Color.RED)  # OK
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

// An overloaded converter only contributes the input types of overloads callable with a single
// positional argument; an overload requiring a second positional arg is ignored.
attrs_testcase!(
    test_attrs_field_overloaded_converter_single_positional,
    r#"
from typing import overload, reveal_type
from attrs import define, field

@overload
def conv(x: int) -> str: ...
@overload
def conv(x: str, y: int) -> str: ...
def conv(x: object, y: int = 0) -> str:
    return str(x)

@define
class C:
    a: str = field(converter=conv)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int) -> None
"#,
);

// An overload requiring a second keyword-only argument is also ignored (it can't be called with
// a single positional arg).
attrs_testcase!(
    test_attrs_field_overloaded_converter_required_kwonly,
    r#"
from typing import overload, reveal_type
from attrs import define, field

@overload
def conv(x: int) -> str: ...
@overload
def conv(x: bytes, *, mode: int) -> str: ...
def conv(x: object, *, mode: int = 0) -> str:
    return str(x)

@define
class C:
    a: str = field(converter=conv)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int) -> None
"#,
);

// A generic-class converter (`list[int]`) applies its type arguments: the `__init__` param
// takes the parameterized constructor's input type, not `Any`.
attrs_testcase!(
    test_attrs_field_generic_class_converter,
    r#"
from typing import assert_type
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list[int])

assert_type(C([1, 2, 3]).xs, list[int])
C(5)  # E: not assignable to parameter `xs`
"#,
);

// The element type of a builtin generic converter is enforced: `list[int]` accepts `Iterable[int]`,
// so a `list[str]` argument is rejected.
attrs_testcase!(
    test_attrs_field_generic_converter_wrong_element,
    r#"
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list[int])

C(["a"])  # E: not assignable to parameter `xs` with type `Iterable[int]`
"#,
);

// A user-defined generic converter applies its type argument directly: `Box[int]`'s `__init__`
// parameter `T` becomes `int`, while the stored attribute keeps the declared `Box[int]`.
attrs_testcase!(
    test_attrs_field_user_generic_converter,
    r#"
from typing import assert_type
from attrs import define, field

class Box[T]:
    def __init__(self, x: T) -> None: ...

@define
class C:
    b: Box[int] = field(converter=Box[int])

assert_type(C(5).b, Box[int])
C("x")  # E: not assignable to parameter `b` with type `int`
"#,
);

// The type argument is substituted into nested positions of the converter's parameter: `Sink[int]`,
// whose `__init__` takes `list[T]`, yields an `__init__` parameter of `list[int]`.
attrs_testcase!(
    test_attrs_field_generic_converter_nested_typevar,
    r#"
from attrs import define, field

class Sink[T]:
    def __init__(self, xs: list[T]) -> None: ...

@define
class C:
    s: Sink[int] = field(converter=Sink[int])

C(5)  # E: not assignable to parameter `s` with type `list[int]`
"#,
);

// A bare (unsubscripted) generic converter still works via the class-object path: `list` promotes
// to `list[Unknown]`, so the `__init__` parameter is `Iterable[Unknown]`.
attrs_testcase!(
    test_attrs_field_bare_generic_converter,
    r#"
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list)

C(5)  # E: not assignable to parameter `xs` with type `Iterable[Unknown]`
"#,
);

// `attr.converters.optional(c)` makes the `__init__` param the inner converter's input type
// unioned with `None`.
attrs_testcase!(
    test_attrs_field_converters_optional,
    r#"
from attrs import define, field
import attr

@define
class C:
    x: int = field(converter=attr.converters.optional(int))

C(None)     # OK: optional converter accepts None
C([1, 2])   # E: not assignable to parameter `x`
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

// A `factory=` callable's return type must be assignable to the field type, just like an
// explicit `default=` value.
attrs_testcase!(
    test_attrs_field_factory_return_type_mismatch,
    r#"
from attrs import define, field

def make_str() -> str:
    return ""

@define
class C:
    x: int = field(factory=make_str)  # E: `str` is not assignable to `int`
"#,
);

// A matching `factory=` return type is accepted.
attrs_testcase!(
    test_attrs_field_factory_return_type_match,
    r#"
from attrs import define, field

def make_int() -> int:
    return 0

@define
class C:
    x: int = field(factory=make_int)
"#,
);

// A `factory=` whose output feeds a `converter=` is checked against the converter's input,
// not the field type, so a "mismatched" factory return is not flagged.
attrs_testcase!(
    test_attrs_field_factory_with_converter_not_checked,
    r#"
from attrs import define, field

def make_str() -> str:
    return ""

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(factory=make_str, converter=to_int)
"#,
);

// Likewise an explicit `default=` value with a `converter=` is the converter's input, so it
// is not checked against the field type.
attrs_testcase!(
    test_attrs_field_default_with_converter_not_checked,
    r#"
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(default="5", converter=to_int)
"#,
);

// A `converter=` that is itself a type constructor (`converter=int`) is supported: the init
// parameter accepts the constructor's input types while the attribute keeps the converted output
// type, and an argument the constructor can't accept is rejected.
attrs_testcase!(
    test_attrs_field_converter_is_constructor,
    r#"
from typing import assert_type
from attrs import define, field

@define
class C:
    x: int = field(default="5", converter=int)

assert_type(C("5").x, int)
C(b"10")
C([1, 2])  # E: not assignable to parameter `x`
"#,
);

// Legacy `attr.ib` accepts a positional `default`, so it is checked against the annotation.
attrs_testcase!(
    test_attrs_attr_ib_positional_default_checked,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib("bad")  # E: `Literal['bad']` is not assignable to `int`
"#,
);

// Next-gen `field` is keyword-only: a positional arg is only an arg-count error and must NOT also
// be treated as a `default` and checked against the annotation (no spurious assignability error).
attrs_testcase!(
    test_attrs_field_positional_not_treated_as_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field("bad")  # E: No matching overload found
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

// A non-default field inherited from one base, ordered after a defaulted field from another,
// is a merge-induced ordering error reported at the subclass definition.
attrs_testcase!(
    test_attrs_inherited_nondefault_after_default,
    r#"
from attrs import define, field

@define
class Base:
    a: int = field(default=5)

@define
class Mixin:
    b: int = field()

@define
class Sub(Mixin, Base):  # E: without a default may not follow
    pass
"#,
);

// A conflict contained within a single base is reported once (on that base); a subclass that
// merely inherits it does NOT re-report it.
attrs_testcase!(
    test_attrs_inherited_conflict_not_reported_on_subclass,
    r#"
from attrs import define, field

@define
class Base:
    a: int = field(default=5)
    b: int = field()  # E: without a default may not follow

@define
class Sub(Base):
    pass
"#,
);

// A required field declared in a class that *inherits* a defaulted field: the conflict
// originates at — and is reported once at — that class; subclasses inheriting it stay silent.
attrs_testcase!(
    test_attrs_inherited_default_local_required_not_repeated,
    r#"
from attrs import define, field

@define
class HasDefault:
    where: int = field(default=0)

@define
class Origin(HasDefault):
    arg: int = field()  # E: without a default may not follow

@define
class Inheritor(Origin):
    pass
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
    x: int = field(default="oops")  # E: `Literal['oops']` is not assignable to `int`
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

// `attr.ib`'s first positional arg is `default`, so positional NOTHING ⇒ required. The NOTHING
// sentinel means "no default", so it must not be checked against the field's declared type.
attrs_testcase!(
    test_attrs_field_nothing_positional_required,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(attr.NOTHING)

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

attrs_testcase!(
    field_validator_decorator_with_explicit_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=2)
    items: list[int] = field(factory=list)

    @x.validator
    def _check_x(self, attribute, value):
        pass

    @items.validator
    def _check_items(self, attribute, value):
        pass

C()  # OK
"#,
);

// The `Any` retype applies only to attrs specifiers: a `@x.default` on a plain class still errors.
attrs_testcase!(
    non_attrs_default_decorator_still_errors,
    r#"
class C:
    x: int = 0

    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

attrs_testcase!(
    field_validator_decorator_multiple,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _a(self, attribute, value):
        pass

    @x.validator
    def _b(self, attribute, value):
        pass

C(1)  # OK: validators are additive
"#,
);

// The `Any` retype is attrs-specific: a stdlib `@dataclass` field keeps its declared type, so
// `@x.default` errors there too.
attrs_testcase!(
    dataclass_field_default_decorator_still_errors,
    r#"
from dataclasses import dataclass, field

@dataclass
class C:
    x: int = field()

    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

attrs_testcase!(
    field_default_and_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @a.default
    def _a(self):
        return 0

    @a.validator
    def _check_a(self, attribute, value):
        pass

C()   # OK
C(1)  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_legacy_attr_ib,
    r#"
import attr

@attr.s
class C:
    x = attr.ib()

    @x.default
    def _x(self):
        return 0

C()   # OK
C(1)  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_legacy_attr_ib_with_type,
    r#"
import attr
from typing import reveal_type

@attr.s
class C:
    x = attr.ib(type=int)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int = ...) -> None
C(1)  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_ordering,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @a.default
    def _a(self):
        return 0

    b: int = field()  # E: without a default may not follow
"#,
);

attrs_testcase!(
    field_default_decorator_kw_only,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(kw_only=True)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, *, x: int = ...) -> None
"#,
);

// The `@a.default` param keeps the field's declared type (`int`), not the in-body specifier `Any`.
attrs_testcase!(
    field_default_decorator_init_signature,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    b: int = field()
    a: int = field()

    @a.default
    def _a(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, b: int, a: int = ...) -> None
"#,
);

attrs_testcase!(
    field_default_decorator_inherited,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

    @x.default
    def _x(self):
        return 0

@define
class Sub(Base):
    y: int = field(default=1)

Sub()   # OK
Sub(0)  # OK
"#,
);

// A subclass override may add a `@x.default`, making the field optional in the subclass only.
attrs_testcase!(
    field_default_decorator_override_in_subclass,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    x: int = field()

    @x.default
    def _x(self):
        return 0

Base()  # E: Missing argument `x`
Sub()   # OK
"#,
);

// A `@x.default` method decorates the `field()` object named in the current class body (see the
// decorator form under attrs "Defaults": https://www.attrs.org/en/stable/init.html#defaults), so a
// subclass re-declares `x` rather than decorating the inherited field; that re-declaration replaces
// (not combines with) the parent's default, so it is not a conflict.
attrs_testcase!(
    field_default_decorator_override_parent_default,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field(default=1)

@define
class Sub(Base):
    x: int = field()

    @x.default
    def _x(self):
        return 0

Sub()  # OK
"#,
);

// `init=False` excludes the field from `__init__` regardless of the `@x.default`.
attrs_testcase!(
    field_default_decorator_init_false,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(init=False)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C) -> None
"#,
);

// Cross-module: a `@x.default` field in another module is still optional in the subclass.
testcase!(
    field_default_decorator_inherited_cross_module,
    {
        let mut env = attrs_env();
        env.add(
            "base",
            r#"
from attrs import define, field

@define
class Base:
    x: int = field()

    @x.default
    def _x(self):
        return 0
"#,
        );
        env
    },
    r#"
from attrs import define, field
from base import Base

@define
class Sub(Base):
    y: int = field(default=1)

Sub()  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_nested_in_control_flow,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    if True:
        @x.default
        def _x(self):
            return 0

C()   # OK
C(1)  # OK
"#,
);

// The decorator scan stops at nested scopes, so `Inner`'s `@x.default` does not make `C.x` optional.
attrs_testcase!(
    field_default_decorator_not_leaked_from_nested_class,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @define
    class Inner:
        x: int = field()

        @x.default
        def _x(self):
            return 0

C()          # E: Missing argument `x`
C.Inner()    # OK
"#,
);

// An undefined name in `@<name>.default` is an unbound-name error and does not make any field optional.
attrs_testcase!(
    field_default_decorator_undefined_name,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @b.default  # E: Could not find name `b`
    def _b(self):
        return 0

C()  # E: Missing argument `a`
"#,
);

// `@x.default` in a subclass cannot target a field inherited from a base: `x` is not in the
// subclass body, so it errors rather than making the inherited field optional.
attrs_testcase!(
    field_default_decorator_targets_inherited_field,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

// `default=` and a `@x.default` method are mutually exclusive (attrs raises `DefaultAlreadySetError`).
attrs_testcase!(
    field_default_decorator_conflicts_with_explicit_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=1)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return 0
"#,
);

// `factory=` is also a default, so it likewise conflicts with a `@x.default` method.
attrs_testcase!(
    field_default_decorator_conflicts_with_factory,
    r#"
from attrs import define, field

@define
class C:
    x: list[int] = field(factory=list)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return []
"#,
);

// `default=attr.NOTHING` means "no default", so a `@x.default` is the sole default, not a conflict.
attrs_testcase!(
    field_default_decorator_with_nothing_default,
    r#"
import attr
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING)

    @x.default
    def _x(self):
        return 0

C()   # OK
C(1)  # OK
"#,
);

// `attr.ib`'s positional default also conflicts with a `@x.default` method.
attrs_testcase!(
    field_default_decorator_conflicts_with_positional_attr_ib,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(5)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return 0
"#,
);

// attrs raises `DefaultAlreadySetError` for a second `@x.default` on the same field.
attrs_testcase!(
    field_default_decorator_duplicate,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: `x` cannot have more than one `@x.default` method

    @x.default
    def _a(self):
        return 0

    @x.default
    def _b(self):
        return 1
"#,
);

// A duplicate `@x.default` with a mismatched return type reports only the duplicate error.
attrs_testcase!(
    field_default_decorator_duplicate_skips_return_type_check,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: `x` cannot have more than one `@x.default` method

    @x.default
    def _a(self) -> str:
        return "a"

    @x.default
    def _b(self) -> str:
        return "b"
"#,
);

// The `@x.default` method's return type must be assignable to the field's declared type.
attrs_testcase!(
    field_default_decorator_return_type_mismatch,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: Return type `str` of the `@x.default` method is not assignable to field `x` of type `int`

    @x.default
    def _x(self) -> str:
        return "oops"
"#,
);

attrs_testcase!(
    field_default_decorator_return_type_match,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self):
        return 0

C()  # OK
"#,
);

// With a converter the default flows through the converter's input type, so the return type is
// not checked against the field type.
attrs_testcase!(
    field_default_decorator_with_converter_not_checked,
    r#"
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(converter=to_int)

    @x.default
    def _x(self):
        return "0"

C()  # OK
"#,
);

// The return-type check resolves through string/forward-ref annotations (PEP 563).
attrs_testcase!(
    field_default_decorator_return_type_mismatch_forward_ref,
    r#"
from __future__ import annotations
from attrs import define, field

@define
class C:
    x: int = field()  # E: Return type `str` of the `@x.default` method is not assignable to field `x` of type `int`

    @x.default
    def _x(self) -> str:
        return "oops"
"#,
);

// The `@x.default` method is called as `meth(self)`; requiring another argument is an error.
attrs_testcase!(
    field_default_decorator_wrong_signature,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, extra):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has required parameters that attrs does not pass
        return 0
"#,
);

// The `@x.validator` method is called as `validator(self, attribute, value)`; fewer params is an error.
attrs_testcase!(
    field_validator_decorator_wrong_signature,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it accepts too few positional parameters
        pass
"#,
);

// `*args` absorbs attrs' call shape for both decorators, so neither signature is flagged.
attrs_testcase!(
    field_decorator_signature_varargs_ok,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, *args):
        return 0

    @x.validator
    def _check(self, *args):
        pass

C(1)  # OK
"#,
);

// `*args` does not satisfy a required parameter: `extra` is still unfilled by attrs' `meth(self)`.
attrs_testcase!(
    field_default_decorator_required_arg_with_varargs,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, extra, *args):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has required parameters that attrs does not pass
        return 0
"#,
);

// A required keyword-only parameter can never be filled by attrs' positional call.
attrs_testcase!(
    field_validator_decorator_required_kwonly,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self, attribute, value, *, k):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it has a required keyword-only parameter that attrs cannot pass
        pass
"#,
);

// The default method, too, cannot be passed a required keyword-only parameter.
attrs_testcase!(
    field_default_decorator_required_kwonly,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, *, k):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has a required keyword-only parameter that attrs cannot pass
        return 0
"#,
);

// attrs passes exactly `(self, attribute, value)`; an extra required positional is unfillable.
attrs_testcase!(
    field_validator_decorator_too_many_required,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self, attribute, value, extra):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it has required parameters that attrs does not pass
        pass
"#,
);

// attrs keeps only the LAST of a duplicated field name, at the last position with the last type
// (desired: `(self: C, y: int, x: str)`). pyrefly keeps the FIRST position and loses the type —
// the root is a binding-phase bug (scope.rs `Static::upsert` re-keys the rebound name to
// `Anywhere`), separate from the `get_dataclass_fields` reorder. Tracked for a follow-up.
attrs_testcase!(
    bug = "Same-class duplicate field keeps first position and loses the override type",
    test_attrs_same_class_duplicate_field,
    r#"
from typing import reveal_type
import attr

@attr.s
class C:
    x: int = attr.ib()
    y: int = attr.ib()
    x: str = attr.ib()  # E: `x` cannot be annotated with `str`, it is already defined with type `int`

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int) -> None
"#,
);

// The eq/order/cmp combination rules apply to field specifiers too, not just the decorator.
attrs_testcase!(
    test_attrs_field_eq_false_order_true,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(eq=False, order=True)  # E: `order` cannot be True when `eq` is False
"#,
);

// Classic `attr.ib` rejects `cmp` mixed with `eq`/`order`.
attrs_testcase!(
    test_attrs_attr_ib_cmp_with_eq,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(cmp=True, eq=True)  # E: Cannot mix `cmp` with `eq` or `order`
"#,
);

// A callable `eq` (a key function) is truthy, so `order=True` alongside it is legal.
attrs_testcase!(
    test_attrs_field_callable_eq_with_order_ok,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(eq=str, order=True)
"#,
);
