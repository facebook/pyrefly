/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;

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
