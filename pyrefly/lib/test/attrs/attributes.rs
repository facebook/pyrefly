/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;
use crate::test::attrs::util::attrs_env;
use crate::testcase;

// attrs names a private attribute's `__init__` parameter without its leading underscores
// (`_private` -> `private`); the attribute itself keeps the underscore.
attrs_testcase!(
    private_attrs,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.s(auto_attribs=True)
class Example:
    _private: str
    public: int

reveal_type(Example.__init__)  # E: revealed type: (self: Example, private: str, public: int) -> None
obj = Example(private="secret", public=42)
assert_type(obj._private, str)
"#,
);

// An explicit `alias=` wins over underscore stripping: the param keeps the given name.
attrs_testcase!(
    test_attrs_private_attr_alias_overrides_stripping,
    r#"
from typing import reveal_type
import attrs

@attrs.define
class C:
    _x: int = attrs.field(alias="custom")

reveal_type(C.__init__)  # E: revealed type: (self: C, custom: int) -> None
"#,
);

// A stripped name that collides with another field is a duplicate-argument error in attrs.
attrs_testcase!(
    test_attrs_private_attr_strip_name_collision,
    r#"
from typing import reveal_type
from attrs import define

@define
class C:
    x: int
    _x: str  # E: collides with `x`

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, _x: str) -> None
"#,
);

// Only LEADING underscores are stripped; trailing underscores are kept (and a name with
// trailing underscores is not name-mangled, so `__both__` keeps its trailing dunder).
attrs_testcase!(
    test_attrs_private_attr_strips_only_leading_underscores,
    r#"
from typing import reveal_type
from attrs import define

@define
class C:
    _lead: int
    trail_: int
    __both__: int

reveal_type(C.__init__)  # E: revealed type: (self: C, lead: int, trail_: int, both__: int) -> None
"#,
);

// Stripping composes with `kw_only=True`: the param is renamed and placed after `*`.
attrs_testcase!(
    test_attrs_private_kw_only_field,
    r#"
from typing import reveal_type
import attrs

@attrs.define
class C:
    _x: int = attrs.field(kw_only=True)

reveal_type(C.__init__)  # E: revealed type: (self: C, *, x: int) -> None
"#,
);

// Stripping composes with a default: the param is renamed and optional, while the attribute
// keeps its underscore.
attrs_testcase!(
    test_attrs_private_attr_with_default,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.s(auto_attribs=True)
class C:
    _x: int = 5

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int = ...) -> None
C()
C(x=1)
assert_type(C()._x, int)
"#,
);

// The stripped name is the only accepted keyword: `_x=` is rejected and there is no `.x`
// attribute, while `._x` keeps its type.
attrs_testcase!(
    test_attrs_private_attr_construction_and_access,
    r#"
from typing import assert_type
import attrs

@attrs.define
class C:
    _x: int

c = C(x=1)
assert_type(c._x, int)
C(_x=1)  # E: Missing argument `x` # E: Unexpected keyword argument `_x`
c.x  # E: Object of class `C` has no attribute `x`
"#,
);

// Stripping applies across a deeper inheritance chain mixing private and public fields:
// privates lose their leading underscore in `__init__` (in MRO order) while public fields
// are unchanged.
attrs_testcase!(
    test_attrs_private_attr_mixed_deep_inheritance,
    r#"
from typing import reveal_type
from attrs import define

@define
class Base:
    _a: int
    b: str

@define
class Mid(Base):
    _c: float

@define
class Sub(Mid):
    _d: int
    e: bool

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, a: int, b: str, c: float, d: int, e: bool) -> None
"#,
);

// Cross-module inheritance: a private field from a base class in another module is still
// stripped in the subclass `__init__`.
testcase!(
    test_attrs_private_attr_inherited_cross_module,
    {
        let mut env = attrs_env();
        env.add(
            "base",
            r#"
from attrs import define

@define
class Base:
    _a: int
"#,
        );
        env
    },
    r#"
from typing import reveal_type
from attrs import define
from base import Base

@define
class Sub(Base):
    _b: int

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, a: int, b: int) -> None
Sub(a=1, b=2)
"#,
);

// A dunder-leading field `__y` is Python name-mangled to `_C__y` before attrs strips leading
// underscores, so the init param is `C__y`.
attrs_testcase!(
    test_attrs_dunder_leading_name_mangling,
    r#"
from typing import reveal_type
import attrs

@attrs.define
class C:
    __y: int

reveal_type(C.__init__)  # E: revealed type: (self: C, C__y: int) -> None
"#,
);

// Name mangling uses the *defining* class, so an inherited dunder-leading field keeps the base's
// mangled init name in the subclass: `Base.__y` stays `Base__y`, not `Sub__y`.
attrs_testcase!(
    test_attrs_dunder_leading_name_mangling_inherited,
    r#"
from typing import reveal_type
import attrs

@attrs.define
class Base:
    __y: int

@attrs.define
class Sub(Base):
    __z: int

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, Base__y: int, Sub__z: int) -> None
"#,
);

// `@attr.s(auto_attribs=True)` opts into annotation-driven fields (classic
// `@attr.s` defaults to `auto_attribs=False`, see the tests below).
attrs_testcase!(
    test_attrs_basic,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s(auto_attribs=True)
class A:
    x: int
    y: int | None = None

reveal_type(A.__init__)  # E: revealed type: (self: A, x: int, y: int | None = ...) -> None

a = A(1)
assert_type(a.x, int)
assert_type(a.y, int | None)
"#,
);

// `@attr.s()` defaults to `auto_attribs=False`: bare annotations are NOT attrs
// fields, so `__init__` takes no parameters and `A(1)` is an error. The
// annotations still describe ordinary attributes.
attrs_testcase!(
    test_attrs_classic_s_default_ignores_bare_annotations,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.s()
class A:
    x: int
    y: int | None = None

reveal_type(A.__init__)  # E: revealed type: (self: A) -> None
A(1)  # E: Expected 0 positional arguments
assert_type(A().x, int)
assert_type(A().y, int | None)
"#,
);

// Explicit `auto_attribs=False` matches the default: only `attr.ib()`-assigned names
// become fields; the bare `y` is not a field.
attrs_testcase!(
    test_attrs_classic_s_explicit_no_auto_attribs,
    r#"
from typing import reveal_type
import attr

@attr.s(auto_attribs=False)
class A:
    x: int = attr.ib()
    y: int

reveal_type(A.__init__)  # E: revealed type: (self: A, x: int) -> None
"#,
);

// `attr.attr` is an alias of `attr.attrib` and must be a recognized field specifier.
// Checked via construction (not the inferred param type) to stay robust to the separate
// converter-inference fix that changes these params from `Unknown` to their annotation.
attrs_testcase!(
    test_attrs_attr_alias_field_specifier,
    r#"
import attr

@attr.s
class A:
    x: int = attr.ib()
    y: int = attr.attr()

A(x=1, y=2)  # `y` (attr.attr) is a field, so it is an init parameter
"#,
);

// An inherited `attr.attr()` field becomes an init parameter on the subclass.
attrs_testcase!(
    test_attrs_attr_alias_field_specifier_inherited,
    r#"
import attr

@attr.s
class Base:
    a: int = attr.attr()

@attr.s
class Sub(Base):
    b: int = attr.ib()

Sub(a=1, b=2)  # inherited `a` (attr.attr) is an init parameter
"#,
);

// `@attr.dataclass` aliases the classic `attrs` stub function (`dataclass = attrs`)
// but is `partial(attrs, auto_attribs=True)`, so unlike `@attr.s` it IS
// annotation-driven: bare annotations are fields. We distinguish it from `@attr.s`
// by the decorator's syntactic name.
attrs_testcase!(
    test_attr_dataclass_is_auto_attribs_true,
    r#"
from typing import assert_type, reveal_type
import attr

@attr.dataclass
class C:
    x: int
    y: int | None = None

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int | None = ...) -> None
c = C(1)
assert_type(c.x, int)
assert_type(c.y, int | None)
"#,
);

attrs_testcase!(
    test_attrs_attrib_pass,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s()
class A:
    x: int = attr.ib()
    y: int | None = attr.ib(None)

a = A(1)
assert_type(a.x, int)
assert_type(a.y, int | None)
"#,
);

attrs_testcase!(
    test_attrs_attrib_init_signature,
    r#"
from typing import assert_type, reveal_type

import attr

@attr.s()
class A:
    x: int = attr.ib()
    y: int | None = attr.ib(None)

reveal_type(A.__init__)  # E: revealed type: (self: A, x: int, y: int | None = ...) -> None
"#,
);

// Classic `@attr.s` (`auto_attribs=False`): an unannotated `attr.ib()` is a field, so it
// becomes an `__init__` parameter without requiring a type annotation. The type is `Any`
// when nothing constrains it, or inferred from `default`.
attrs_testcase!(
    test_attrs_unannotated_attrib_ok,
    r#"
from typing import reveal_type
import attr
@attr.s()
class A:
    x = attr.ib()  # !E: type annotation
    y = attr.ib(default=0)  # !E: type annotation

reveal_type(A.__init__)  # E: revealed type: (self: A, x: Any, y: int = ...) -> None
A()  # E: Missing argument `x`
"#,
);

// Under `auto_attribs=True`, attrs collects fields from annotations, so an unannotated
// `attr.ib()` is not a valid field and attrs raises `UnannotatedAttributeError` at runtime.
attrs_testcase!(
    test_attrs_unannotated_attrib_auto_attribs,
    r#"
import attr
@attr.s(auto_attribs=True)
class A:
    x = attr.ib()  # E: needs a type annotation
"#,
);

// The error is localized to the unannotated field: an annotated sibling is unaffected and is
// the only `__init__` parameter (the unannotated field is dropped, matching attrs).
attrs_testcase!(
    test_attrs_unannotated_attrib_auto_attribs_mixed,
    r#"
from typing import reveal_type
import attr
@attr.s(auto_attribs=True)
class A:
    x: int
    y = attr.ib()  # E: needs a type annotation

reveal_type(A.__init__)  # E: revealed type: (self: A, x: int) -> None
"#,
);

// The error is reported once, at the base class where the unannotated field is declared, and
// not duplicated on a subclass that inherits it.
attrs_testcase!(
    test_attrs_unannotated_attrib_auto_attribs_inherited,
    r#"
import attr
@attr.s(auto_attribs=True)
class Base:
    x = attr.ib()  # E: needs a type annotation

@attr.s(auto_attribs=True)
class Sub(Base):
    y: int
"#,
);

// `attr.ib(type=T)` supplies the field type when there is no annotation, so the field
// becomes a typed `__init__` parameter (annotation wins if both are given).
attrs_testcase!(
    test_attrs_attrib_type_keyword,
    r#"
from typing import reveal_type
import attr

@attr.s
class C:
    x = attr.ib(type=int)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int) -> None
C("hello")  # E: not assignable to parameter `x`
"#,
);

// Next-gen `field()` is the opposite of `attr.ib`: attrs documents that type checkers ignore
// its `type=` metadata, so the field stays untyped (`Any`) rather than `int`.
attrs_testcase!(
    test_attrs_field_type_keyword_ignored,
    r#"
from typing import reveal_type
import attr, attrs

@attr.s
class C:
    x = attrs.field(type=int)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: Any) -> None
C("anything")  # `x` is untyped, so any argument is accepted
"#,
);

// A `type=`-typed field declared in a base class is typed in the subclass `__init__`.
attrs_testcase!(
    test_attrs_attrib_type_keyword_inherited,
    r#"
from typing import reveal_type
import attr

@attr.s
class Base:
    a = attr.ib(type=int)

@attr.s
class Sub(Base):
    b = attr.ib(type=str)

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, a: int, b: str) -> None
"#,
);

// An unannotated `attr.ib()` inherited from a classic-attrs base is a parameter of the
// subclass `__init__`, so the field's metadata is resolved against its defining class.
attrs_testcase!(
    test_attrs_unannotated_attrib_inherited,
    r#"
import attr

@attr.s
class Base:
    a = attr.ib()

@attr.s
class Sub(Base):
    b = attr.ib(default=0)

Sub(1)
Sub()  # E: Missing argument `a`
"#,
);

// attrs raises `ValueError` at runtime if a field has both a type annotation and a
// `type=` argument. Matching types are used so the only error is the conflict itself.
attrs_testcase!(
    test_attrs_attrib_type_and_annotation_conflict,
    r#"
import attr

@attr.s
class C:
    x: int = attr.ib(type=int)  # E: both a type annotation and a `type` argument
"#,
);

// The conflict applies to the modern API too: next-gen `field()` under `@define`.
attrs_testcase!(
    test_attrs_field_type_and_annotation_conflict_define,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(type=int)  # E: both a type annotation and a `type` argument
"#,
);

// Even when the annotation and `type=` disagree, the conflict is reported once: the specifier's
// widened return is no longer assignment-checked against the annotation.
attrs_testcase!(
    test_attrs_attrib_type_and_annotation_conflict_mismatch,
    r#"
import attr

@attr.s
class C:
    x: int = attr.ib(type=str)  # E: both a type annotation and a `type` argument
"#,
);

// A `type=` keyword on a non-specifier call is not a field declaration, so no conflict error
// fires even with an annotation (attrs only inspects `attr.ib`/`field` results).
attrs_testcase!(
    test_attrs_type_keyword_on_non_specifier_call,
    r#"
import attr

def helper(type: int) -> int:
    return 0

@attr.s(auto_attribs=True)
class C:
    x: int = helper(type=5)
"#,
);

// Scoping: `type=` handling is attrs-only. A stdlib `@dataclass` is unaffected
// (`dataclasses.field` has no `type=`), so `x` does not become a typed field.
attrs_testcase!(
    test_attrs_type_keyword_dataclass_unaffected,
    r#"
from dataclasses import dataclass, field
from typing import reveal_type

@dataclass
class C:
    x = field(type=int)  # E: has no type annotation # E: No matching overload

reveal_type(C.__init__)  # E: revealed type: (self: C) -> None
"#,
);
