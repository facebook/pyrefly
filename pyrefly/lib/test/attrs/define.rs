/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;
use crate::test::attrs::util::attrs_env;
use crate::testcase;

// DECORATOR DETECTION
//
// Coverage of every class-level attrs decorator entry point. All are marked with PEP 681
// `@dataclass_transform(...)` in attrs' stubs, so pyrefly routes them through its generic
// dataclass synthesis path. dataclass_transform alone can't express attrs' per-decorator
// `auto_attribs` rules, so an attrs-specific layer resolves which assignments are fields:
//   - `@define` / `@mutable` / `@frozen` (+ `attr.`/`attrs.` forms): `auto_attribs=None`,
//     i.e. annotation-driven unless a bare `attr.ib()`/`field()` assignment forces
//     `auto_attribs=False`
//   - `@attr.s` / `@attr.s()` / `@attr.attrs` / `@attr.attributes`: `auto_attribs=False`
//     (only `attr.ib()`/`field()` assignments are fields; bare annotations ignored)
//   - `@attr.s(auto_attribs=True)` / `@attr.dataclass`: `auto_attribs=True`
//   - per-decorator default keywords: order_default (classic), frozen_default, no-order (define)
//
// NOT handled (bug-marked tests below):
//   - `field()` / `attr.ib()` specifier params -> `Unknown` (converter-param leak)

// `@define`: fields come from annotations (auto_attribs is implicit).
attrs_testcase!(
    test_attrs_define_basic,
    r#"
from typing import assert_type, reveal_type
from attrs import define

@define
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None

c = C(1)
assert_type(c.x, int)
assert_type(c.y, int | None)
"#,
);

// `@attrs.define` namespace form.
attrs_testcase!(
    test_attrs_namespace_define_basic,
    r#"
from typing import assert_type, reveal_type
import attrs

@attrs.define
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None

c = C(1)
assert_type(c.x, int)
assert_type(c.y, int | None)
"#,
);

// `@attr.define` namespace form.
attrs_testcase!(
    test_attr_namespace_define_basic,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.define
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None

c = C(1)
assert_type(c.x, int)
assert_type(c.y, int | None)
"#,
);

// Classic `@attr.s(auto_attribs=True)`.
attrs_testcase!(
    test_attrs_classic_s_basic,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.s(auto_attribs=True)
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None

c = C(1)
assert_type(c.x, int)
assert_type(c.y, int | None)
"#,
);

// `field()` with an explicit annotation and no `converter`: the synthesized `__init__`
// param types are the declared annotations. (A converter is only read from an explicit
// `converter=` argument, never from the field specifier's signature.)
attrs_testcase!(
    test_attrs_define_field_init_signature,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field()
    y: int = field(default=0)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int = ...) -> None
"#,
);

// `@frozen` (frozen_default=True): attributes are read-only.
attrs_testcase!(
    test_attrs_frozen_basic,
    r#"
from typing import assert_type, reveal_type
from attrs import frozen

@frozen
class C:
    x: int

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int) -> None
c = C(1)
assert_type(c.x, int)
c.x = 2  # E: Cannot set field `x`
"#,
);

// `@attr.frozen` / `@attrs.frozen` namespace forms.
attrs_testcase!(
    test_attr_namespace_frozen_basic,
    r#"
import attr
import attrs

@attr.frozen
class A:
    x: int

@attrs.frozen
class B:
    x: int

A(1).x = 2  # E: Cannot set field `x`
B(1).x = 2  # E: Cannot set field `x`
"#,
);

// A frozen attrs subclass of a non-frozen base is rejected at the declaration (like stdlib),
// not at each write. The own frozen field `b` is still read-only.
attrs_testcase!(
    test_attrs_frozen_subclass_of_non_frozen_base,
    r#"
import attrs

@attrs.define
class P:
    a: int

@attrs.frozen
class C(P):  # E: Cannot inherit frozen dataclass `C` from non-frozen dataclass `P`
    b: int = 0

c = C(1)
c.a = 5
c.b = 6  # E: Cannot set field `b`
"#,
);

// Same as above, but the non-frozen base lives in another module.
testcase!(
    test_attrs_frozen_subclass_of_non_frozen_base_cross_module,
    {
        let mut env = attrs_env();
        env.add(
            "base",
            r#"
import attrs

@attrs.define
class P:
    a: int
"#,
        );
        env
    },
    r#"
import attrs
from base import P

@attrs.frozen
class C(P):  # E: Cannot inherit frozen dataclass `C` from non-frozen dataclass `P`
    b: int = 0

c = C(1)
c.a = 5
c.b = 6  # E: Cannot set field `b`
"#,
);

// `@mutable` is an alias of `@define` (attributes writable). Covers all namespace forms.
attrs_testcase!(
    test_attrs_mutable_basic,
    r#"
from typing import assert_type, reveal_type
from attrs import mutable
import attr
import attrs

@mutable
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None
c = C(1)
assert_type(c.x, int)
c.x = 2  # OK: mutable

@attr.mutable
class A:
    x: int

@attrs.mutable
class B:
    x: int

assert_type(A(1).x, int)
assert_type(B(1).x, int)
"#,
);

// `@attr.attrs` / `@attr.attributes` are aliases of `@attr.s`.
attrs_testcase!(
    test_attr_classic_aliases,
    r#"
from typing import assert_type
import attr

@attr.attrs(auto_attribs=True)
class A:
    x: int

@attr.attributes(auto_attribs=True)
class B:
    x: int

assert_type(A(1).x, int)
assert_type(B(1).x, int)
"#,
);

// A classic `attr.ib()` field with an explicit annotation and no `converter`: the
// `__init__` param type is the declared annotation.
attrs_testcase!(
    test_attrs_classic_s_no_auto_attribs,
    r#"
from typing import reveal_type
import attr

@attr.s
class C:
    x: int = attr.ib()

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int) -> None
"#,
);

attrs_testcase!(
    test_attrs_classic_s_no_auto_attribs_ignores_annotations,
    r#"
from typing import reveal_type
import attr

@attr.s()
class C:
    x: int
    y: int

reveal_type(C.__init__)  # E: revealed type: (self: C) -> None
"#,
);

// Classic `@attr.s` sets `order_default=True`, so ordering methods (`__lt__`, etc.) are
// synthesized.
attrs_testcase!(
    test_attrs_classic_order,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int

C(1) < C(2)  # OK: order_default=True synthesizes __lt__
"#,
);

// `@define` does NOT set `order_default`, so ordering methods are not synthesized.
attrs_testcase!(
    test_attrs_define_no_order,
    r#"
from attrs import define

@define
class C:
    x: int

C(1) < C(2)  # E: `<` is not supported
"#,
);

// DECORATOR KEYWORDS
//
// Decorator-level keywords (init / frozen / slots / kw_only / match_args / order /
// unsafe_hash) and frozen-subclass propagation. These map onto the standard dataclass
// keywords, so pyrefly handles them through the generic dataclass_transform path.

// `@attr.dataclass`, a fourth class-level entry point (stub-aliased to classic `attrs`).
attrs_testcase!(
    test_attr_dataclass_basic,
    r#"
from typing import assert_type
import attr

@attr.dataclass
class C:
    x: int

assert_type(C(1).x, int)
"#,
);

// `@attr.dataclass` defaults to `auto_attribs=True` by name (no auto-detect fallback), so an
// unannotated specifier needs a type annotation.
attrs_testcase!(
    test_attr_dataclass_unannotated_needs_annotation,
    r#"
import attr

@attr.dataclass
class C:
    x = attr.ib()  # E: needs a type annotation
"#,
);

// `init=False` suppresses `__init__` synthesis.
attrs_testcase!(
    test_attrs_define_init_false,
    r#"
from attrs import define

@define(init=False)
class C:
    x: int

C()    # OK: no synthesized __init__, falls back to object.__init__
C(1)   # E: Expected 0 positional arguments
"#,
);

// With `init=False`, attrs still synthesizes the field initializer as `__attrs_init__`, so a
// hand-written `__init__` can delegate to it.
attrs_testcase!(
    test_attrs_define_init_false_attrs_init,
    r#"
from typing import reveal_type
from attrs import define

@define(init=False)
class C:
    x: int

reveal_type(C.__attrs_init__)  # E: revealed type: (self: C, x: int) -> None
"#,
);

// Explicit `frozen=True` keyword on `@define`.
attrs_testcase!(
    test_attrs_define_frozen_kwarg,
    r#"
from attrs import define

@define(frozen=True)
class C:
    x: int

C(1).x = 2  # E: Cannot set field `x`
"#,
);

// `kw_only=True` at the class level makes all fields keyword-only in `__init__`.
attrs_testcase!(
    test_attrs_define_kw_only,
    r#"
from attrs import define

@define(kw_only=True)
class C:
    x: int

C(x=1)  # OK
C(1)    # E: Expected argument `x` to be passed by name
"#,
);

// `order=True` keyword on `@define` enables ordering methods.
attrs_testcase!(
    test_attrs_define_order_kwarg,
    r#"
from attrs import define

@define(order=True)
class C:
    x: int

C(1) < C(2)  # OK
"#,
);

// `match_args` controls `__match_args__` synthesis (default True for `@define`).
attrs_testcase!(
    test_attrs_define_match_args,
    r#"
from typing import reveal_type
from attrs import define

@define
class C:
    x: int
    y: int

reveal_type(C.__match_args__)  # E: revealed type: tuple[Literal['x'], Literal['y']]
"#,
);

// `slots=True` keyword is accepted; field synthesis still works.
attrs_testcase!(
    test_attrs_define_slots,
    r#"
from typing import assert_type
from attrs import define

@define(slots=True)
class C:
    x: int

assert_type(C(1).x, int)
"#,
);

// `unsafe_hash=True` keeps the class hashable.
attrs_testcase!(
    test_attrs_define_unsafe_hash,
    r#"
from attrs import define

@define(unsafe_hash=True)
class C:
    x: int

hash(C(1))  # OK
"#,
);

// `hash=True` is attrs' deprecated alias for `unsafe_hash=True`. Without it `@define`
// (eq=True, frozen=False) would set `__hash__ = None`, making the class unhashable.
attrs_testcase!(
    test_attrs_define_hash_alias,
    r#"
from typing import Hashable
from attrs import define

def f(x: Hashable) -> None: ...

@define(hash=True)
class C:
    x: int

f(C(1))  # OK
"#,
);

// The alias also applies to the classic `@attr.s` decorator.
attrs_testcase!(
    test_attrs_classic_hash_alias,
    r#"
from typing import Hashable
import attr

def f(x: Hashable) -> None: ...

@attr.s(hash=True)
class C:
    x = attr.ib()

f(C(1))  # OK
"#,
);

// `hash=False` overrides the `eq`-driven default: `__hash__` is left inherited (hashable)
// rather than set to `None`.
attrs_testcase!(
    test_attrs_define_hash_false_inherits,
    r#"
from typing import Hashable
from attrs import define

def f(x: Hashable) -> None: ...

@define(hash=False)
class C:
    x: int

f(C(1))  # OK
"#,
);

// `unsafe_hash=` wins over the deprecated `hash=`: with an unhashable base, `unsafe_hash=True`
// still synthesizes `__hash__`, so the class is hashable.
attrs_testcase!(
    test_attrs_unsafe_hash_overrides_hash,
    r#"
from typing import Hashable
from attrs import define

def f(x: Hashable) -> None: ...

class Base:
    __hash__ = None

@define(unsafe_hash=True, hash=False)
class C(Base):
    x: int

f(C(1))  # OK
"#,
);

// Field-level `hash=` is accepted; class hashability follows the class-level decision.
attrs_testcase!(
    test_attrs_field_hash_accepted,
    r#"
from typing import Hashable
from attrs import define, field

def f(x: Hashable) -> None: ...

@define(hash=True)
class C:
    x: int = field(hash=False)

f(C(1))  # OK
"#,
);

// Frozen propagates to subclasses: a plain subclass of a `@frozen` class inherits the
// synthesized frozen `__setattr__`, so attribute assignment is still rejected.
attrs_testcase!(
    test_attrs_frozen_subclass,
    r#"
from attrs import frozen

@frozen
class Base:
    x: int

class Sub(Base):
    pass

Sub(1).x = 2  # E: Cannot set field `x`
"#,
);

// `@define` uses `auto_attribs=None`: attrs falls back to `auto_attribs=False` when a field
// is assigned a bare `field()`/`attr.ib()` with no annotation, so the field is still collected.
attrs_testcase!(
    test_attrs_define_bare_field,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x = field()

reveal_type(C.__init__)  # E: revealed type: (self: C, x: Any) -> None
"#,
);

// Mixing a bare annotation with an unannotated `field()` makes attrs fall back to
// `auto_attribs=False`, so only the `field()` assignment is a field; the bare `a` is an
// annotation-only declaration (not a field, and unset at runtime).
attrs_testcase!(
    test_attrs_define_mixed_bare_field_and_annotation,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    a: int
    b = field()

reveal_type(C.__init__)  # E: revealed type: (self: C, b: Any) -> None
"#,
);

// `auto_attribs` is resolved from each class's own body, so a subclass and its base can use
// different modes; fields are still inherited across the boundary (matches attrs runtime).
attrs_testcase!(
    test_attrs_define_auto_attribs_resolved_per_class,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class Base:
    x = field()        # bare specifier -> Base is auto_attribs=False

@define
class Sub(Base):
    y: int             # annotation -> Sub is auto_attribs=True

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, x: Any, y: int) -> None

@define
class Base2:
    a: int             # annotation -> Base2 is auto_attribs=True

@define
class Sub2(Base2):
    b = field()        # bare specifier -> Sub2 is auto_attribs=False

reveal_type(Sub2.__init__)  # E: revealed type: (self: Sub2, a: int, b: Any) -> None
"#,
);

// A subclass that re-declares an inherited field relocates it to the redefinition position,
// matching attrs (a redefined field moves to its newest declaration site).
attrs_testcase!(
    test_attrs_subclass_override_reorders,
    r#"
from typing import reveal_type
from attrs import define

@define
class Base:
    x: int
    y: str

@define
class Sub(Base):
    z: bool
    x: int  # redeclaring x relocates it after y, z

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, y: str, z: bool, x: int) -> None
"#,
);

// The relocated field uses the override type. Changing a read-write field's type is independently
// flagged (attribute invariance), but the field still moves and `__init__` reflects the new type.
attrs_testcase!(
    test_attrs_subclass_override_changes_type,
    r#"
from typing import reveal_type
from attrs import define

@define
class Base:
    x: int
    y: str

@define
class Sub(Base):
    z: bool
    x: float  # E: not consistent with `int`

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, y: str, z: bool, x: float) -> None
"#,
);

// The corrected order propagates transitively: a grandchild re-declaring a grandparent field
// relocates it to the grandchild position.
attrs_testcase!(
    test_attrs_multilevel_override_reorders,
    r#"
from typing import reveal_type
from attrs import define

@define
class A:
    a: int
    b: int

@define
class B(A):
    c: int

@define
class C(B):
    a: int  # re-declare grandparent field a

reveal_type(C.__init__)  # E: revealed type: (self: C, b: int, c: int, a: int) -> None
"#,
);

// EQ / ORDER / CMP KEYWORD VALIDATION (decorator site)
//
// attrs raises `ValueError` at class creation for two combinations: `order=True` with `eq=False`
// (ordering needs equality), and `cmp` mixed with `eq`/`order` (`cmp` is the legacy alias).

// `order=True` requires `eq` to not be False.
attrs_testcase!(
    test_attrs_decorator_eq_false_order_true,
    r#"
import attr

@attr.s(eq=False, order=True)  # E: `order` cannot be True when `eq` is False
class C:
    x = attr.ib()
"#,
);

// Same rule on the next-gen `@define` path.
attrs_testcase!(
    test_attrs_decorator_define_eq_false_order_true,
    r#"
from attrs import define

@define(eq=False, order=True)  # E: `order` cannot be True when `eq` is False
class C:
    x: int
"#,
);

// `cmp` cannot be combined with `eq`.
attrs_testcase!(
    test_attrs_decorator_cmp_with_eq,
    r#"
import attr

@attr.s(cmp=True, eq=True)  # E: Cannot mix `cmp` with `eq` or `order`
class C:
    x = attr.ib()
"#,
);

// `cmp` cannot be combined with `order` either.
attrs_testcase!(
    test_attrs_decorator_cmp_with_order,
    r#"
import attr

@attr.s(cmp=False, order=True)  # E: Cannot mix `cmp` with `eq` or `order`
class C:
    x = attr.ib()
"#,
);

// Legal combinations must NOT error: `eq` defaulting True with `order=True`; `eq=False` with
// `order` omitted (order mirrors eq); both False; and `cmp` on its own.
attrs_testcase!(
    test_attrs_decorator_eq_order_cmp_legal,
    r#"
import attr
from attrs import define

@attr.s(order=True)
class A:
    x = attr.ib()

@define(eq=False)
class B:
    x: int

@attr.s(eq=False, order=False)
class C:
    x = attr.ib()

@attr.s(cmp=True)
class D:
    x = attr.ib()
"#,
);

// attrs treats an explicit `None` like an omitted argument, so `cmp=None` does not conflict.
attrs_testcase!(
    test_attrs_decorator_cmp_none_with_eq_ok,
    r#"
import attr

@attr.s(cmp=None, eq=True)
class C:
    x = attr.ib()
"#,
);

// `cmp` is the legacy alias setting both `eq` and `order`: `cmp=False` disables ordering even
// though classic `@attr.s` enables it by default.
attrs_testcase!(
    test_attrs_cmp_false_disables_order,
    r#"
import attr

@attr.s(auto_attribs=True, cmp=False)
class C:
    x: int

C(1) < C(2)  # E: `<` is not supported
"#,
);

// `cmp=True` is the same alias the other way: it enables both `eq` and `order`.
attrs_testcase!(
    test_attrs_cmp_true_enables_order,
    r#"
import attr

@attr.s(auto_attribs=True, cmp=True)
class C:
    x: int

C(1) < C(2)  # OK: cmp=True enables ordering
"#,
);

// An explicit `cmp=None` is treated like an omitted argument (not as `cmp=False`), so the
// per-decorator default applies — classic `@attr.s` still enables ordering.
attrs_testcase!(
    test_attrs_cmp_none_is_omitted,
    r#"
import attr

@attr.s(auto_attribs=True, cmp=None)
class C:
    x: int

C(1) < C(2)  # OK: cmp=None falls back to the order default (True for classic `@attr.s`)
"#,
);

// ATTR.FIELDS: the result stays `Any` (it exposes fields by name, which a tuple can't model); we
// only reject non-attrs class arguments.

// `Any` result supports both indexing and by-name access.
attrs_testcase!(
    test_attrs_fields_returns_any,
    r#"
from typing import reveal_type
import attr

@attr.define
class C:
    x: int
    y: str

reveal_type(attr.fields(C))     # E: revealed type: Any
attr.fields(C).x
attr.fields(C)[0]
"#,
);

attrs_testcase!(
    test_attrs_fields_non_attrs_class,
    r#"
import attr

class NotAttrs:
    x: int

attr.fields(NotAttrs)  # E: is not an attrs class
"#,
);

// A dataclass has dataclass metadata of a non-attrs `kind` (distinct path from the plain class).
attrs_testcase!(
    test_attrs_fields_dataclass_rejected,
    r#"
import attr
from dataclasses import dataclass

@dataclass
class D:
    x: int

attr.fields(D)  # E: is not an attrs class
"#,
);

// `type[AttrsInstance]` (the canonical "any attrs class" annotation) must be accepted.
attrs_testcase!(
    test_attrs_fields_attrs_instance_param,
    r#"
import attr
from attr import AttrsInstance

def f(cls: type[AttrsInstance]) -> None:
    attr.fields(cls)
"#,
);

// `attrs.has` narrows to `type[AttrsInstance]` via `TypeGuard`.
attrs_testcase!(
    test_attrs_fields_has_narrowing,
    r#"
import attrs

def f(cls: type) -> None:
    if not attrs.has(cls):
        return
    attrs.fields(cls)
"#,
);

attrs_testcase!(
    test_attrs_fields_type_value,
    r#"
import attr

@attr.define
class C:
    x: int

def f(cls: type[C]) -> None:
    attr.fields(cls)
"#,
);

// ATTR.FIELDS_DICT: returns an ordered name -> `Attribute[T]` mapping, modeled as an anonymous
// TypedDict so each field recovers its precise `Attribute[t]` on subscript.

attrs_testcase!(
    test_attrs_fields_dict_returns_dict,
    r#"
from typing import assert_type
import attr

@attr.define
class C:
    x: int
    y: str

d = attr.fields_dict(C)
assert_type(d["x"], attr.Attribute[int])
assert_type(d["y"], attr.Attribute[str])
"#,
);

attrs_testcase!(
    test_attrs_fields_dict_non_attrs_class,
    r#"
import attr

class NotAttrs:
    x: int

attr.fields_dict(NotAttrs)  # E: `fields_dict()` is not an attrs class
"#,
);

// Inherited fields appear, and a generic class substitutes its type argument.
attrs_testcase!(
    test_attrs_fields_dict_inheritance_generic,
    r#"
from typing import assert_type
import attr

@attr.define
class Base[T]:
    x: T

@attr.define
class Sub(Base[int]):
    y: str

d = attr.fields_dict(Sub)
assert_type(d["x"], attr.Attribute[int])
assert_type(d["y"], attr.Attribute[str])
"#,
);

// Recognition keys off the function's origin, not the import style: a `from attr import` works.
attrs_testcase!(
    test_attrs_fields_dict_from_import,
    r#"
from attr import define, fields_dict
from typing import assert_type
import attr

@define
class C:
    x: int

assert_type(fields_dict(C)["x"], attr.Attribute[int])
"#,
);

// Over the 100-field cap, we fall back to the stub's `dict`: every field degrades to
// `Attribute[Any]`.
attrs_testcase!(
    test_attrs_fields_dict_over_cap_falls_back_to_dict,
    r#"
from typing import Any, assert_type
import attr

@attr.define
class Huge:
    f0: int; f1: int; f2: int; f3: int; f4: int; f5: int; f6: int; f7: int
    f8: int; f9: int; f10: int; f11: int; f12: int; f13: int; f14: int; f15: int
    f16: int; f17: int; f18: int; f19: int; f20: int; f21: int; f22: int; f23: int
    f24: int; f25: int; f26: int; f27: int; f28: int; f29: int; f30: int; f31: int
    f32: int; f33: int; f34: int; f35: int; f36: int; f37: int; f38: int; f39: int
    f40: int; f41: int; f42: int; f43: int; f44: int; f45: int; f46: int; f47: int
    f48: int; f49: int; f50: int; f51: int; f52: int; f53: int; f54: int; f55: int
    f56: int; f57: int; f58: int; f59: int; f60: int; f61: int; f62: int; f63: int
    f64: int; f65: int; f66: int; f67: int; f68: int; f69: int; f70: int; f71: int
    f72: int; f73: int; f74: int; f75: int; f76: int; f77: int; f78: int; f79: int
    f80: int; f81: int; f82: int; f83: int; f84: int; f85: int; f86: int; f87: int
    f88: int; f89: int; f90: int; f91: int; f92: int; f93: int; f94: int; f95: int
    f96: int; f97: int; f98: int; f99: int; f100: str

d = attr.fields_dict(Huge)
assert_type(d["f0"], attr.Attribute[Any])
assert_type(d["f100"], attr.Attribute[Any])
"#,
);

// Field-precise across inheritance and generic substitution: a generic base specialized in a
// subclass yields the exact `Attribute[T]` per field.
attrs_testcase!(
    test_attrs_fields_dict_wide_generic_inheritance,
    r#"
from typing import assert_type
import attr

@attr.define
class Base[T]:
    head: T
    a0: int; a1: int; a2: int; a3: int; a4: int; a5: int; a6: int; a7: int
    a8: int; a9: int; a10: int; a11: int; a12: int; a13: int; a14: int; a15: int
    a16: int; a17: int; a18: int; a19: int; a20: int

@attr.define
class Sub(Base[str]):
    tail: bytes

d = attr.fields_dict(Sub)
assert_type(d["head"], attr.Attribute[str])
assert_type(d["a20"], attr.Attribute[int])
assert_type(d["tail"], attr.Attribute[bytes])
"#,
);

// Field-precise for the classic `@attr.s(auto_attribs=True)` form too, not just `@define`.
attrs_testcase!(
    test_attrs_fields_dict_wide_classic_auto_attribs,
    r#"
from typing import assert_type
import attr

@attr.s(auto_attribs=True)
class Wide:
    a0: int; a1: int; a2: int; a3: int; a4: int; a5: int; a6: int; a7: int
    a8: int; a9: int; a10: int; a11: int; a12: int; a13: int; a14: int; a15: int
    a16: int; a17: int; a18: int; a19: int; a20: int
    last: str

d = attr.fields_dict(Wide)
assert_type(d["a0"], attr.Attribute[int])
assert_type(d["last"], attr.Attribute[str])
"#,
);

// ON_SETATTR
//
// `on_setattr=setters.frozen` makes attributes immutable (attrs raises FrozenAttributeError) without
// the other effects of a fully `frozen` class (no __hash__ change, no frozen-inheritance rule).

attrs_testcase!(
    test_attrs_on_setattr_frozen_class_level,
    r#"
from attr import define, setters

@define(on_setattr=setters.frozen)
class C:
    x: int
    y: str

c = C(1, "a")
_ = c.x
c.x = 2    # E: Cannot set field `x`
c.y = "b"  # E: Cannot set field `y`
"#,
);

attrs_testcase!(
    test_attrs_on_setattr_no_op_writable,
    r#"
from attr import define, setters

@define(on_setattr=setters.NO_OP)
class C:
    x: int

C(1).x = 2  # OK
"#,
);

// A per-field `on_setattr` overrides the class-level frozen-all default, so a field declared with
// `setters.NO_OP` stays writable.
attrs_testcase!(
    test_attrs_field_on_setattr_overrides_class_frozen,
    r#"
from attr import define, field, setters

@define(on_setattr=setters.frozen)
class C:
    x: int
    y: int = field(on_setattr=setters.NO_OP)

c = C(1, 2)
c.x = 5  # E: Cannot set field `x`
c.y = 5  # OK
"#,
);

// `setters.frozen` inside a list of hooks still freezes the field (attrs runs them as a pipe).
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_in_list,
    r#"
from attr import define, field, setters

@define
class C:
    x: int = field(on_setattr=[setters.validate, setters.frozen])
    y: int = field(on_setattr=[setters.validate])

c = C(1, 2)
c.x = 5  # E: Cannot set field `x`
c.y = 5  # OK: no `frozen` hook
"#,
);

// `setters.frozen` inside a `setters.pipe(...)` composition also freezes the field.
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_in_pipe,
    r#"
from attr import define, field, setters

@define
class C:
    x: int = field(on_setattr=setters.pipe(setters.validate, setters.frozen))

C(1).x = 5  # E: Cannot set field `x`
"#,
);

// `on_setattr=setters.frozen` is not full frozen-ness: a frozen-all subclass of a non-frozen base
// must NOT raise the frozen/non-frozen inheritance error, but its own fields are still read-only.
attrs_testcase!(
    test_attrs_on_setattr_frozen_not_inheritance_error,
    r#"
from attr import define, setters

@define
class Base:
    x: int

@define(on_setattr=setters.frozen)
class C(Base):
    y: int

c = C(1, 2)
c.y = 3  # E: Cannot set field `y`
"#,
);

// EVOLVE
//
// `attr.evolve`/`attrs.evolve` copy an instance with changes; the kwargs are validated against
// the class fields like `dataclasses.replace`, and all fields are optional.

attrs_testcase!(
    test_attrs_evolve_basic,
    r#"
from typing import assert_type
import attrs

@attrs.frozen
class Point:
    x: int
    y: int

p = Point(1, 2)
assert_type(attrs.evolve(p, x=5), Point)
attrs.evolve(p)
attrs.evolve(p, x="hello")     # E: not assignable to parameter `x`
attrs.evolve(p, z=3)           # E: Unexpected keyword argument `z`
attrs.evolve(p, nonexistent=4)  # E: Unexpected keyword argument `nonexistent`
"#,
);

// ASSOC
//
// The deprecated `attr.assoc` keys on actual attribute names (no init-alias renaming) and includes
// `init=False` fields, unlike `evolve`'s constructor-alias, init-only semantics.

attrs_testcase!(
    test_attrs_assoc_basic,
    r#"
from typing import assert_type
import attr

@attr.define
class C:
    x: int
    y: int

c = C(1, 2)
assert_type(attr.assoc(c, x=5), C)
attr.assoc(c)
attr.assoc(c, x="bad")    # E: not assignable to parameter `x`
attr.assoc(c, z=3)        # E: Unexpected keyword argument `z`
"#,
);

// `assoc` keys on the attribute name `_x`, whereas `evolve` strips it to the constructor alias `x`.
attrs_testcase!(
    test_attrs_assoc_private_attribute,
    r#"
import attr

@attr.define
class C:
    _x: int

c = C(1)
attr.assoc(c, _x=2)
attr.assoc(c, x=2)        # E: Unexpected keyword argument `x`
attr.evolve(c, x=2)
attr.evolve(c, _x=2)      # E: Unexpected keyword argument `_x`
"#,
);

// `init=False` fields are not constructor params, so `evolve` rejects them while `assoc` accepts them.
attrs_testcase!(
    test_attrs_assoc_init_false_field,
    r#"
import attr

@attr.define
class C:
    x: int
    y: int = attr.field(init=False, default=0)

c = C(1)
attr.assoc(c, y=5)
attr.evolve(c, y=5)       # E: Unexpected keyword argument `y`
"#,
);

// `attr.evolve` on a non-attrs instance is rejected (runtime `NotAnAttrsClassError`), unlike
// `dataclasses.replace` whose stub permits any value.
attrs_testcase!(
    test_attrs_evolve_non_attrs_rejected,
    r#"
import attr

class NotAttrs:
    x: int

attr.evolve(NotAttrs())  # E: is not an attrs class
"#,
);

// A plain stdlib `@dataclass` instance is also not an attrs class: `attr.evolve` rejects it
// even though `dataclasses.replace` would accept it.
attrs_testcase!(
    test_attrs_evolve_plain_dataclass_rejected,
    r#"
import attr
from dataclasses import dataclass

@dataclass
class D:
    x: int

attr.evolve(D(1))  # E: is not an attrs class
"#,
);

// The attrs-only restriction must not leak into `dataclasses.replace`, which still accepts a
// plain `@dataclass`; and `attr.evolve` still works on a real attrs class.
attrs_testcase!(
    test_attrs_evolve_vs_replace_dataclass,
    r#"
import attr
from dataclasses import dataclass, replace

@dataclass
class D:
    x: int

@attr.define
class A:
    x: int

attr.evolve(A(1), x=2)   # OK: attrs class
replace(D(1), x=2)       # OK: `replace` accepts a plain dataclass
attr.evolve(A(1), y=2)   # E: Unexpected keyword argument `y`
"#,
);

// In a union, `attr.evolve` flags the non-attrs member while still checking the attrs member.
attrs_testcase!(
    test_attrs_evolve_union_member_rejected,
    r#"
import attr
from dataclasses import dataclass

@attr.define
class A:
    x: int

@dataclass
class D:
    x: int

def f(o: A | D) -> None:
    attr.evolve(o, x=2)  # E: is not an attrs class
"#,
);

// Best-effort: we only flag concrete non-attrs `ClassType`s, so a non-class instance like a
// TypedDict is left to the (untyped) stub rather than rejected here.
attrs_testcase!(
    test_attrs_evolve_non_class_instance_not_flagged,
    r#"
import attr
from typing import TypedDict

class TD(TypedDict):
    x: int

def f(d: TD) -> None:
    attr.evolve(d)
"#,
);

// `Any` and type variables could resolve to an attrs class at runtime, so `attr.evolve` must
// not reject them.
attrs_testcase!(
    test_attrs_evolve_gradual_not_rejected,
    r#"
import attr
from typing import Any

def f(x: Any) -> None:
    attr.evolve(x)  # OK: `Any` could be an attrs instance

def g[T](y: T) -> None:
    attr.evolve(y)  # OK: a type variable could be an attrs instance
"#,
);

// Inherited fields can be evolved; unknown ones still error.
attrs_testcase!(
    test_attrs_evolve_inheritance,
    r#"
import attrs

@attrs.define
class Base:
    x: int

@attrs.define
class Sub(Base):
    y: str

s = Sub(1, "a")
attrs.evolve(s, x=2, y="b")
attrs.evolve(s, z=3)  # E: Unexpected keyword argument `z`
"#,
);

// Private fields are matched by their stripped init name (`_x` -> `x`).
attrs_testcase!(
    test_attrs_evolve_private_field,
    r#"
import attrs

@attrs.define
class C:
    _x: int

c = C(1)
attrs.evolve(c, x=2)
attrs.evolve(c, _x=2)  # E: Unexpected keyword argument `_x`
"#,
);

// A dunder-leading field is name-mangled with its *defining* class, so an inherited `Base.__y` is
// evolved as `Base__y` (not `Sub__y`) on a subclass instance — matching attrs at runtime.
attrs_testcase!(
    test_attrs_evolve_inherited_mangled_private_field,
    r#"
import attrs

@attrs.define
class Base:
    __y: int

@attrs.define
class Sub(Base):
    __z: int

s = Sub(1, 2)
attrs.evolve(s, Base__y=10, Sub__z=20)
attrs.evolve(s, Sub__y=10)  # E: Unexpected keyword argument `Sub__y`
"#,
);
