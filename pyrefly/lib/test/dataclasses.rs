/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_def,
    r#"
from typing import assert_type
import dataclasses
@dataclasses.dataclass
class Data:
    x: int
    y: str
assert_type(Data, type[Data])
    "#,
);

testcase!(
    test_enum_dataclass_rejected,
    r#"
from dataclasses import dataclass
import dataclasses
from enum import Enum

class Good(Enum):
    RED = 1

@dataclass
class Bad1(Enum):  # E: Cannot apply `@dataclass` to Enum `Bad1`
    RED = 1

@dataclasses.dataclass
class Bad2(Enum):  # E: Cannot apply `@dataclass` to Enum `Bad2`
    RED = 1

@dataclass()
class Bad3(Enum):  # E: Cannot apply `@dataclass` to Enum `Bad3`
    RED = 1
    "#,
);

testcase!(
    test_kw_only_sentinel_deep_inheritance,
    r#"
from dataclasses import dataclass, KW_ONLY

@dataclass
class A:
    _: KW_ONLY
    a: int = 0

@dataclass
class B(A):
    b: int = 1

@dataclass
class C(B):
    _: KW_ONLY
    c: int = 2

@dataclass
class D(C):
    d: int = 3

D()
D(4)
D(4, 5)
D(4, 5, 6) # E: Expected 2 positional arguments, got 3 in function `D.__init__`
    "#,
);

testcase!(
    test_fields,
    r#"
from typing import assert_type
import dataclasses
@dataclasses.dataclass
class Data:
    x: int
    y: str
def f(d: Data):
    assert_type(d.x, int)
    assert_type(d.y, str)
    "#,
);

testcase!(
    test_generic,
    r#"
from typing import assert_type
import dataclasses
@dataclasses.dataclass
class Data[T]:
    x: T
def f(d: Data[int]):
    assert_type(d.x, int)
assert_type(Data(x=0), Data[int])
Data[int](x=0)  # OK
Data[int](x="")  # E: Argument `Literal['']` is not assignable to parameter `x` with type `int` in function `Data.__init__`
    "#,
);

testcase!(
    test_construction,
    r#"
import dataclasses
@dataclasses.dataclass
class Data:
    x: int
    y: str
Data(0, "1")  # OK
Data(0, 1)  # E: Argument `Literal[1]` is not assignable to parameter `y` with type `str`
    "#,
);

testcase!(
    test_replace,
    r#"
from dataclasses import dataclass, replace

@dataclass
class Foo:
    x: int
    y: str

f = Foo(1, "a")

replace(f, x="wrong")  # E: Argument `Literal['wrong']` is not assignable to parameter `x` with type `int` in function `Foo.__replace__`
replace(f, z=3)  # E: Unexpected keyword argument `z`
    "#,
);

testcase!(
    test_copy_replace,
    TestEnv::new_with_version(PythonVersion::new(3, 13, 0)),
    r#"
import copy
from copy import replace
from dataclasses import dataclass
from typing import assert_type

@dataclass
class Foo:
    x: int
    y: str

f = Foo(1, "a")

assert_type(copy.replace(f, x=2), Foo)
replace(f, y="b")
copy.replace(f, x="wrong")  # E: Argument `Literal['wrong']` is not assignable to parameter `x` with type `int` in function `Foo.__replace__`
replace(f, z=3)  # E: Unexpected keyword argument `z`
    "#,
);

testcase!(
    test_replace_initvar_default,
    r#"
from dataclasses import dataclass, field, InitVar, replace

@dataclass
class WithInitVarDefault:
    x: int
    y: InitVar[str] = "ok"

w = WithInitVarDefault(0)
replace(w)
replace(w, y="new")
    "#,
);

testcase!(
    test_replace_initvar_required,
    r#"
from dataclasses import dataclass, InitVar, replace

@dataclass
class Foo:
    x: int
    y: InitVar[int]

f = Foo(1, 2)

replace(f)  # E: Missing argument `y`
    "#,
);

testcase!(
    test_replace_positional_args_rejected,
    r#"
from dataclasses import dataclass, replace

@dataclass
class Foo:
    x: int
    y: str

f = Foo(1, "a")

replace(f, "extra")  # E: Expected 0 positional arguments, got 1
    "#,
);

testcase!(
    test_replace_init_false_field_rejected,
    r#"
from dataclasses import dataclass, field, replace

@dataclass
class WithInitFalse:
    x: int
    y: int = field(init=False, default=5)

g = WithInitFalse(1)

replace(g, y=10)  # E: Unexpected keyword argument `y`
    "#,
);

testcase!(
    test_replace_classvar_rejected,
    r#"
from dataclasses import dataclass, replace
from typing import ClassVar

@dataclass
class Config:
    limit: int
    MAX_ID: ClassVar[int] = 100

c = Config(10)
replace(c, limit=20)
replace(c, MAX_ID=200) # E: Unexpected keyword argument `MAX_ID`
    "#,
);

testcase!(
    test_replace_union_mixed_dataclass,
    r#"
from dataclasses import dataclass, replace
from typing import Union

@dataclass
class Foo:
    x: int

class Bar:
    x: int

def f(obj: Union[Foo, Bar]):
    replace(obj, x=0)  # E: `Bar` is not assignable to upper bound `DataclassInstance`
    replace(obj, x="oops")  # E: `Bar` is not assignable to upper bound `DataclassInstance`  # E: Argument `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_replace_union_two_dataclasses_rejects_bad_kw,
    r#"
from dataclasses import dataclass, replace
from typing import Union

@dataclass
class A:
    x: int

@dataclass
class B:
    y: int

def f(obj: Union[A, B]):
    replace(obj, z=1)  # E: Unexpected keyword argument `z` in function `A.__replace__`  # E: Unexpected keyword argument `z` in function `B.__replace__`
    "#,
);

testcase!(
    test_replace_union_two_dataclasses_rejects_kw_not_in_all_members,
    r#"
from dataclasses import dataclass, replace
from typing import Union

@dataclass
class A:
    x: int

@dataclass
class B:
    y: int

def f(obj: Union[A, B]):
    replace(obj, x=1)  # E: Unexpected keyword argument `x`
    "#,
);

testcase!(
    test_replace_union_two_dataclasses_accepts_shared_kw,
    r#"
from dataclasses import dataclass, replace
from typing import Union

@dataclass
class A:
    x: int

@dataclass
class B:
    x: int
    y: str

def f(obj: Union[A, B]):
    replace(obj, x=1)
    "#,
);

testcase!(
    test_replace_starred_args_rejected,
    r#"
from dataclasses import dataclass, replace

@dataclass
class Foo:
    x: int
    y: int

foo = Foo(1, 2)

replace(foo, *())
replace(foo, **{"x": "bad"})  # E: Argument `str` is not assignable to parameter `x` with type `int`
replace(foo, **{"z": 0})  # E: Unexpected keyword argument `z`
    "#,
);

testcase!(
    test_replace_rejects_obj_keyword,
    r#"
from dataclasses import dataclass, replace

@dataclass
class Foo:
    x: int

foo = Foo(1)

replace(foo, obj=foo)  # E: Unexpected keyword argument `obj`
    "#,
);

testcase!(
    test_replace_generic_consistency,
    r#"
from dataclasses import dataclass, replace
from typing import TypeVar, Generic, assert_type

T = TypeVar("T")

@dataclass
class Box(Generic[T]):
    item: T

b = Box(item=1)
assert_type(replace(b, item=2), Box[int])
replace(b, item="wrong")  # E: Argument `Literal['wrong']` is not assignable to parameter `item` with type `int`
    "#,
);

testcase!(
    test_replace_any_object_allows_any_keywords,
    r#"
from dataclasses import replace
from typing import Any

def f(obj: Any):
    replace(obj, z=1)
    replace(obj, **{"z": 2})
    "#,
);

testcase!(
    test_replace_union_with_dataclass_and_any,
    r#"
from dataclasses import dataclass, replace
from typing import Any, assert_type

@dataclass
class Foo:
    x: int

def f(obj: Foo | Any):
    replace(obj, x="oops")  # E: Argument `Literal['oops']` is not assignable to parameter `x` with type `int`
    assert_type(replace(obj, x=0), Foo | Any)
    "#,
);

testcase!(
    test_replace_treats_dataclass_transform_as_dataclass,
    r#"
from dataclasses import replace
from typing import dataclass_transform

@dataclass_transform()
def my_dc(cls):
    return cls

@my_dc
class Model:
    x: int
    y: str

    def __init__(self, x: int, y: str) -> None: ...

m = Model(1, "a")

replace(m, x=2)
replace(m, x="oops")  # E: Argument `Literal['oops']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_inheritance,
    r#"
import dataclasses

@dataclasses.dataclass
class A:
    w: int

class B(A):
    x: str
# B is not decorated as a dataclass, so w is the only dataclass field
B(w=0)

@dataclasses.dataclass
class C(B):
    y: bytes
# C is decorated as a dataclass again, so w and y are the dataclass fields
C(w=0, y=b"1")

@dataclasses.dataclass
class D(C):
    z: float
# Make sure we get the parameters in the right order when there are multiple @dataclass bases
D(0, b"1", 2.0)
    "#,
);

testcase!(
    test_asdict,
    r#"
import dataclasses
from typing import assert_type

@dataclasses.dataclass
class A:
    x: int
    y: str
    items: list[int]

d = dataclasses.asdict(A(3, "a", [1]))
# Subscripting the synthesized TypedDict recovers each field's precise type...
assert_type(d["x"], int)
assert_type(d["y"], str)
assert_type(d["items"], list[int])
# ...preserved all the way down: the list element type is not widened to Any.
assert_type(d["items"][0], int)
# The whole value coerces to dict[str, <field-type union>].
assert_type(d, dict[str, int | str | list[int]])
    "#,
);

testcase!(
    test_asdict_generic,
    r#"
import dataclasses
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")

@dataclasses.dataclass
class Box(Generic[T]):
    item: T
    items: list[T]

def f(b: Box[int]):
    # The type argument is substituted into the field types, including nested ones.
    d = dataclasses.asdict(b)
    assert_type(d["item"], int)
    assert_type(d["items"], list[int])
    assert_type(d["items"][0], int)
    "#,
);

testcase!(
    test_asdict_union,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class A:
    x: int

@dataclasses.dataclass
class B:
    x: str

def f(ab: A | B):
    # A union argument is not a single dataclass type, so the anonymous-TypedDict
    # special-case does not fire and we fall back to the declared signature, which
    # widens the value type to Any.
    d = dataclasses.asdict(ab)
    assert_type(d, dict[str, Any])
    assert_type(d["x"], Any)
    "#,
);

testcase!(
    test_asdict_dict_factory,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class A:
    x: int

def factory(items: list[tuple[str, Any]]) -> list[str]:
    return []

# The `dict_factory=` overload returns the factory's result type, so the
# anonymous-TypedDict special-case does not fire.
result = dataclasses.asdict(A(x=3), dict_factory=factory)
assert_type(result, list[str])
assert_type(result[0], str)
    "#,
);

testcase!(
    test_asdict_nested,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class Inner:
    a: int

@dataclasses.dataclass
class Outer:
    inner: Inner
    items: list[Inner]
    n: int

d = dataclasses.asdict(Outer(Inner(1), [Inner(2)], 3))
# asdict recurses at runtime: nested dataclass instances become dicts, including
# those inside containers, so their types collapse to dict[str, Any]. A
# non-recursive implementation would instead type d["inner"] as `Inner`.
assert_type(d["inner"], dict[str, Any])
assert_type(d["items"], list[dict[str, Any]])
assert_type(d["n"], int)
assert_type(d, dict[str, dict[str, Any] | int | list[dict[str, Any]]])
    "#,
);

testcase!(
    test_asdict_recursive,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class One:
    x: "One"
    y: int

def f(o: One):
    # `One` is not a runtime value type -- asdict(o)["x"] is a dict -- so the
    # self-referential field collapses to dict[str, Any] instead of expanding
    # forever.
    d = dataclasses.asdict(o)
    assert_type(d["x"], dict[str, Any])
    assert_type(d["y"], int)
    assert_type(d, dict[str, dict[str, Any] | int])
    "#,
);

testcase!(
    test_asdict_mutually_recursive,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class A:
    b: "B"
    n: int

@dataclasses.dataclass
class B:
    a: "A"
    m: str

def f(o: A):
    # Mutual recursion: each nested dataclass collapses to dict[str, Any].
    d = dataclasses.asdict(o)
    assert_type(d["b"], dict[str, Any])
    assert_type(d["n"], int)
    assert_type(d, dict[str, dict[str, Any] | int])
    "#,
);

testcase!(
    test_asdict_only_dataclasses_replaced,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class Inner:
    a: int

class Plain:  # not a dataclass: must be left untouched
    pass

@dataclasses.dataclass
class C:
    p: Plain
    inner: Inner

def f(o: C):
    # Only dataclass types are rewritten; an ordinary class field is preserved.
    d = dataclasses.asdict(o)
    assert_type(d["p"], Plain)
    assert_type(d["inner"], dict[str, Any])
    assert_type(d, dict[str, Plain | dict[str, Any]])
    "#,
);

testcase!(
    test_asdict_dataclass_in_containers,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class Inner:
    a: int

@dataclasses.dataclass
class C:
    md: dict[str, Inner]
    tp: tuple[Inner, int]
    un: Inner | int
    op: Inner | None
    deep: list[dict[str, list[Inner]]]

def f(o: C):
    # The replacement traverses into every position of the type tree.
    d = dataclasses.asdict(o)
    assert_type(d["md"], dict[str, dict[str, Any]])
    assert_type(d["tp"], tuple[dict[str, Any], int])
    assert_type(d["un"], dict[str, Any] | int)
    assert_type(d["op"], dict[str, Any] | None)
    assert_type(d["deep"], list[dict[str, list[dict[str, Any]]]])
    "#,
);

testcase!(
    test_asdict_union_of_dataclasses_collapses,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class A:
    a: int

@dataclasses.dataclass
class B:
    b: str

@dataclasses.dataclass
class C:
    field: A | B

def f(o: C):
    # Distinct dataclasses in a union all collapse to the same dict[str, Any], so
    # they merge into a single value type.
    d = dataclasses.asdict(o)
    assert_type(d["field"], dict[str, Any])
    assert_type(d, dict[str, dict[str, Any]])
    "#,
);

testcase!(
    test_asdict_field_selection,
    r#"
import dataclasses
from typing import ClassVar, assert_type

@dataclasses.dataclass
class A:
    x: int
    cv: ClassVar[float] = 0.0  # ClassVar: not an instance field, excluded
    no_init: str = dataclasses.field(init=False, default="")  # init=False: still included
    iv: dataclasses.InitVar[bytes] = b""  # InitVar: not an instance field, excluded

    def __post_init__(self, iv: bytes) -> None:
        pass

d = dataclasses.asdict(A(0))
assert_type(d["x"], int)
assert_type(d["no_init"], str)
# Only the instance fields x and no_init appear: the value union is int | str, so
# the excluded ClassVar(float) and InitVar(bytes) do not leak into the type.
assert_type(d, dict[str, int | str])
    "#,
);

testcase!(
    test_asdict_inherited,
    r#"
import dataclasses
from typing import assert_type

@dataclasses.dataclass
class Base:
    a: int

@dataclasses.dataclass
class Derived(Base):
    b: str

d = dataclasses.asdict(Derived(1, "x"))
assert_type(d["a"], int)  # inherited field
assert_type(d["b"], str)  # own field
assert_type(d, dict[str, int | str])
    "#,
);

testcase!(
    test_asdict_optional,
    r#"
import dataclasses
from typing import Optional, assert_type

@dataclasses.dataclass
class A:
    x: Optional[int]
    y: int | None

d = dataclasses.asdict(A(1, 2))
assert_type(d["x"], int | None)
assert_type(d["y"], int | None)
assert_type(d, dict[str, int | None])
    "#,
);

testcase!(
    test_asdict_many_same_type_fields,
    r#"
import dataclasses
from typing import Any, assert_type

# 25 fields: over the field-count cap, so we fall back to dict[str, Any]. The cap is
# on field count (decided before building the TypedDict), so we lose precision here
# even though every field shares a single type and could have stayed int.
@dataclasses.dataclass
class Many:
    a0: int
    a1: int
    a2: int
    a3: int
    a4: int
    a5: int
    a6: int
    a7: int
    a8: int
    a9: int
    a10: int
    a11: int
    a12: int
    a13: int
    a14: int
    a15: int
    a16: int
    a17: int
    a18: int
    a19: int
    a20: int
    a21: int
    a22: int
    a23: int
    a24: int

def f(o: Many):
    d = dataclasses.asdict(o)
    assert_type(d["a0"], Any)
    assert_type(d, dict[str, Any])
    "#,
);

testcase!(
    test_asdict_field_count_limit_within,
    r#"
import dataclasses
from typing import assert_type

# Exactly 20 fields: at the cap, so the special-case still fires and every field
# keeps its precise type.
@dataclasses.dataclass
class Within:
    a0: int
    a1: str
    a2: bytes
    a3: float
    a4: bool
    a5: complex
    a6: list[int]
    a7: list[str]
    a8: dict[str, int]
    a9: set[int]
    a10: frozenset[str]
    a11: tuple[int, str]
    a12: list[bytes]
    a13: dict[int, str]
    a14: set[str]
    a15: list[float]
    a16: list[bool]
    a17: dict[str, bytes]
    a18: set[bytes]
    a19: memoryview

def f(o: Within):
    d = dataclasses.asdict(o)
    assert_type(d["a0"], int)
    assert_type(d["a19"], memoryview)
    "#,
);

testcase!(
    test_asdict_field_count_limit_exceeded,
    r#"
import dataclasses
from typing import Any, assert_type

# 21 fields: over the cap, so we fall back to `dict[str, Any]` rather than
# synthesize an unwieldy 21-member TypedDict.
@dataclasses.dataclass
class Exceeded:
    a0: int
    a1: str
    a2: bytes
    a3: float
    a4: bool
    a5: complex
    a6: list[int]
    a7: list[str]
    a8: dict[str, int]
    a9: set[int]
    a10: frozenset[str]
    a11: tuple[int, str]
    a12: list[bytes]
    a13: dict[int, str]
    a14: set[str]
    a15: list[float]
    a16: list[bool]
    a17: dict[str, bytes]
    a18: set[bytes]
    a19: memoryview
    a20: range

def f(o: Exceeded):
    d = dataclasses.asdict(o)
    assert_type(d["a0"], Any)
    assert_type(d, dict[str, Any])
    "#,
);

testcase!(
    test_asdict_mixed_union,
    r#"
import dataclasses
from typing import Any, assert_type

@dataclasses.dataclass
class A:
    x: int

def f(a: A | int):
    # A union argument is not a single dataclass type, so it falls back to the
    # declared signature (which also flags the non-dataclass member as a bad argument).
    d = dataclasses.asdict(a)  # E: is not assignable to parameter
    assert_type(d, dict[str, Any])
    "#,
);

testcase!(
    test_asdict_not_dataclass,
    r#"
import dataclasses
dataclasses.asdict(42)  # E: is not assignable to parameter
    "#,
);

testcase!(
    test_duplicate_field,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    x: int
    y: float
@dataclasses.dataclass
class B(A):
    x: str # E:  Class member `B.x` overrides parent class `A` in an inconsistent manner
# Overwriting x doesn't change the param order but does change its type
B('0', 1.0)  # OK
B(0, 1.0)  # E: Argument `Literal[0]` is not assignable to parameter `x` with type `str`
    "#,
);

// A property that shadows an inherited field without re-annotating it must not redefine
// the field's type or keywords (e.g. `init=False`).

testcase!(
    test_property_override_init_false_field,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    foo: int = dataclasses.field(init=False)
@dataclasses.dataclass
class B(A):
    @property
    def foo(self) -> int:  # E: Class member `B.foo` overrides parent class `A` in an inconsistent manner
        return 1
# `foo` is `init=False`, inherited from `A`: it is not a constructor parameter.
B()  # OK
B(foo=2)  # E: Unexpected keyword argument `foo`
    "#,
);

testcase!(
    test_classvar_override_field,
    r#"
import dataclasses
from typing import ClassVar
@dataclasses.dataclass
class A:
    foo: int
@dataclasses.dataclass
class B(A):
    foo: ClassVar[int] = 0  # E: ClassVar `B.foo` overrides instance variable of the same name in parent class `A`
# A `ClassVar` override removes `foo` from the constructor entirely.
B()  # OK
B(foo=1)  # E: Unexpected keyword argument `foo`
    "#,
);

testcase!(
    test_property_override_field_diamond_mro,
    r#"
import dataclasses
@dataclasses.dataclass
class Base:
    foo: int = dataclasses.field(init=False)
@dataclasses.dataclass
class Mixin:
    @property
    def foo(self) -> int:
        return 1
@dataclasses.dataclass
class C(Mixin, Base):
    pass
# Resolution walks the MRO past `Mixin`'s property to `Base`'s `init=False` field.
C()  # OK
C(foo=2)  # E: Unexpected keyword argument `foo`
    "#,
);

testcase!(
    test_property_override_attribute_access_uses_property,
    r#"
import dataclasses
from typing import assert_type
@dataclasses.dataclass
class A:
    foo: int = dataclasses.field(init=False)
@dataclasses.dataclass
class B(A):
    @property
    def foo(self) -> str:  # E: Class member `B.foo` overrides parent class `A` in an inconsistent manner
        return "x"
def f(b: B):
    # Attribute access still resolves to the property getter, not the inherited field.
    assert_type(b.foo, str)
    "#,
);

testcase!(
    test_property_override_preserves_match_args,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    foo: int = 0
    bar: int = 0
@dataclasses.dataclass
class B(A):
    @property
    def foo(self) -> int:  # E: Class member `B.foo` overrides parent class `A` in an inconsistent manner
        return 1
def f(b: B):
    # `foo` is still a field, so it stays in `__match_args__` alongside `bar`;
    # this irrefutable two-element pattern must match without error.
    match b:
        case B(x, y):
            pass
    "#,
);

testcase!(
    test_property_override_replace_uses_field_type,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    foo: int = 0
@dataclasses.dataclass
class B(A):
    @property
    def foo(self) -> int:  # E: Class member `B.foo` overrides parent class `A` in an inconsistent manner
        return 1
def f(b: B):
    # The synthesized `__replace__` also uses the inherited field's `int` type.
    dataclasses.replace(b, foo=10)
    dataclasses.replace(b, foo="bad")  # E: Argument `Literal['bad']` is not assignable to parameter `foo` with type `int`
    "#,
);

testcase!(
    test_field_overrides_parent_property,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    @property
    def foo(self) -> int:
        return 1
@dataclasses.dataclass
class B(A):
    foo: int = 5
# The reverse direction: an annotated field DOES redefine the inherited property,
# so `foo` becomes a real constructor parameter with a default.
B()  # OK
B(5)  # OK
B("x")  # E: Argument `Literal['x']` is not assignable to parameter `foo` with type `int`
    "#,
);

testcase!(
    test_inherit_from_multiple_dataclasses,
    r#"
import dataclasses
@dataclasses.dataclass
class A:
    x: int
@dataclasses.dataclass
class B:
    y: str

class C(B, A):
    pass
C(y="0")  # First base (B) wins

@dataclasses.dataclass
class D(B, A):
    z: float
D(0, "1", 2.0)
    "#,
);

testcase!(
    test_inherit_from_generic_dataclass,
    r#"
import dataclasses
@dataclasses.dataclass
class A[T]:
    x: T
@dataclasses.dataclass
class B(A[int]):
    y: str
B(x=0, y="1")  # OK
B(x="0", y="1")  # E: Argument `Literal['0']` is not assignable to parameter `x` with type `int` in function `B.__init__`
    "#,
);

testcase!(
    test_decorate_with_call_return,
    r#"
from dataclasses import dataclass
@dataclass()
class C:
    x: int
C(x=0)  # OK
C(x='0')  # E: Argument `Literal['0']` is not assignable to parameter `x` with type `int` in function `C.__init__`
    "#,
);

testcase!(
    test_init_already_defined,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int
    def __init__(self):
        self.x = 42
C()  # OK
C(x=0)  # E: Unexpected keyword argument
    "#,
);

testcase!(
    test_init_false,
    r#"
from dataclasses import dataclass
@dataclass(init=False)
class C:
    x: int = 0
C()  # OK
C(x=0)  # E: Unexpected keyword argument
    "#,
);

testcase!(
    test_with_methods,
    r#"
from typing import assert_type, Any, Literal
from dataclasses import dataclass
@dataclass
class C:
    x: int = 0
    def m(self) -> int: return self.x
c = C()  # Ok
assert_type(c.m(), int) # Ok
a: Any = ...
C(m=a)  # E: Unexpected keyword argument `m`
assert_type(c.__match_args__, tuple[Literal['x']])  # Ok
    "#,
);

testcase!(
    bug = "TODO: consider erroring on a plain unannotated assignment like `y = 3`",
    test_unannotated_attribute,
    r#"
import dataclasses
@dataclasses.dataclass
class C:
    # Not annotating a field with value dataclasses.field(...) is a runtime error.
    x = dataclasses.field()  # E: `x` is a dataclass field but has no type annotation
    # This is confusing and likely indicative of a programming error; consider erroring on this, too.
    y = 3
    "#,
);

testcase!(
    test_frozen,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int

@dataclass(frozen=True)
class D:
    x: int

def f(c: C, d: D):
    c.x = 0
    d.x = 0  # E: Cannot set field `x`
    "#,
);

testcase!(
    test_match_args,
    r#"
from typing import assert_type, Literal
from dataclasses import dataclass
@dataclass
class C_has_match_args_default:
    x: int
@dataclass(match_args=True)
class C_has_match_args_explicit:
    x: int
@dataclass(match_args=False)
class C_no_match_args:
    x: int
assert_type(C_has_match_args_default.__match_args__, tuple[Literal['x']])
assert_type(C_has_match_args_explicit.__match_args__, tuple[Literal['x']])
C_no_match_args.__match_args__ # E: no class attribute `__match_args__`
    "#,
);

testcase!(
    test_match_args_no_overwrite,
    r#"
from typing import assert_type
from dataclasses import dataclass
@dataclass(match_args=True)
class C:
    __match_args__ = ()
    x: int
assert_type(C.__match_args__, tuple[()])
    "#,
);

testcase!(
    test_kw_only_arg,
    r#"
from typing import assert_type
from dataclasses import dataclass
@dataclass(kw_only=True)
class C:
    x: int
C(x=0)  # OK
C(0)  # E: Expected argument `x` to be passed by name
assert_type(C.__match_args__, tuple[()])
    "#,
);

testcase!(
    test_kw_only_sentinel,
    r#"
from typing import assert_type, Literal
import dataclasses
@dataclasses.dataclass
class C:
    x: int
    _: dataclasses.KW_ONLY
    y: str
C(0, y="1")  # OK
C(x=0, y="1")  # OK
C(0, "1")  # E: Expected argument `y` to be passed by name
assert_type(C.__match_args__, tuple[Literal["x"]])
    "#,
);

testcase!(
    test_order,
    r#"
from dataclasses import dataclass
@dataclass
class D1:
    x: int
def f(d: D1, e: D1):
    if d < e: ...  # E: `<` is not supported between `D1` and `D1`
    if d == e: ...  # OK: `==` and `!=` never error regardless

@dataclass(order=True)
class D2:
    x: int
@dataclass(order=True)
class D3:
    x: int
def f(d: D2, e: D2, f: D3):
    if d < e: ...  # OK
    if e < f: ...  # E: `<` is not supported between `D2` and `D3`\n  Argument `D3` is not assignable to parameter `other` with type `D2`
    if e != f: ...  # OK: `==` and `!=` never error regardless
    "#,
);

testcase!(
    test_call_comparison_unbound_with_named_args,
    r#"
from dataclasses import dataclass
@dataclass(order=True)
class D: pass
D.__lt__(self=D(), other=D())
    "#,
);

testcase!(
    test_bad_keyword,
    r#"
from dataclasses import dataclass
@dataclass(flibbertigibbet=True)  # E: Unexpected keyword argument `flibbertigibbet`
class C:
    pass
    "#,
);

testcase!(
    test_dataclasses_field_with_init_flag,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(init=False)
    y: str
C(y="")  # OK
C(x=0, y="")  # E: Unexpected keyword argument `x`
    "#,
);

testcase!(
    test_dataclass_field_with_default_factory,
    r#"
from dataclasses import dataclass, field
@dataclass(frozen=True)
class C:
    x: list[str] = field(default_factory=list)
    "#,
);

// `default` and `default_factory` are mutually exclusive at runtime; typeshed's
// `dataclasses.field` overloads already reject passing both, so no extra check is needed.
testcase!(
    test_dataclass_field_default_and_default_factory_conflict,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(default=1, default_factory=int)  # E: not assignable to parameter `default_factory`
    "#,
);

testcase!(
    test_default,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int = 0
C()  # OK
C(x=1)  # OK
    "#,
);

testcase!(
    test_field_is_not_default,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field()
C()  # E: Missing argument `x`
    "#,
);

testcase!(
    test_field_kw_only,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(kw_only=True)
C(1)  # E: Expected argument `x` to be passed by name
C(x=1)  # OK
    "#,
);

testcase!(
    test_field_default,
    r#"
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class C1:
    x: int = field(default=0)
C1()  # OK
C1(x=1)  # OK

factory: Callable[[], int] = lambda: 0

@dataclass
class C2:
    x: int = field(default_factory=factory)
C2()  # OK
C2(x=1)  # OK

@dataclass
class C3:
    x: int = field(default="oops")  # E: `str` is not assignable to `int`
    y: str = field(default_factory=factory)  # E: `int` is not assignable to `str`
    "#,
);

testcase!(
    test_classvar,
    r#"
from typing import ClassVar
from dataclasses import dataclass
@dataclass
class C:
    x: ClassVar[int] = 0
C()  # OK
C(x=1)  # E: Unexpected keyword argument `x`
    "#,
);

testcase!(
    test_inherit_classvar,
    r#"
from typing import ClassVar
from dataclasses import dataclass
@dataclass
class C:
    x: ClassVar[int]
@dataclass
class D(C):
    x = 0
D()  # OK
D(x=1)  # E: Unexpected keyword argument `x`
    "#,
);

testcase!(
    test_bare_classvar,
    r#"
from typing import ClassVar
import dataclasses
@dataclasses.dataclass
class C:
    replace: ClassVar = dataclasses.replace
C()
    "#,
);

testcase!(
    test_frozen_classvar_class_assignment,
    r#"
import dataclasses
from typing import ClassVar

@dataclasses.dataclass(frozen=True)
class C:
    x: ClassVar[bool] = True

    def set_x(self) -> None:
        self.__class__.x = False
    "#,
);

testcase!(
    test_hashable,
    r#"
from typing import Hashable
from dataclasses import dataclass

class Unhashable:
    __hash__ = None

def f(x: Hashable):
    pass

# When eq=frozen=True, __hash__ is implicitly created
@dataclass(eq=True, frozen=True)
class D1(Unhashable):
    pass
f(D1())  # OK

# When eq=True, frozen=False, __hash__ is set to None
@dataclass(eq=True, frozen=False)
class D2:
    pass
f(D2())  # E: Argument `D2` is not assignable to parameter `x` with type `Hashable`

# When eq=False, __hash__ is untouched
@dataclass(eq=False)
class D3:
    pass
@dataclass(eq=False)
class D4(Unhashable):
    pass
f(D3())  # OK
f(D4())  # E: Argument `D4` is not assignable to parameter `x` with type `Hashable`

# unsafe_hash=True forces __hash__ to be created
@dataclass(eq=False, unsafe_hash=True)
class D5(Unhashable):
    pass
f(D5())  # OK
    "#,
);

testcase!(
    test_bad_mro,
    r#"
from dataclasses import dataclass

@dataclass
class A:
  x: int

@dataclass
class B:
  pass

@dataclass
class C(A, B, A):  # E: nonlinearizable inheritance chain
  pass

def f(c: C):
    return c.x  # E: `C` has no attribute `x`
    "#,
);

testcase!(
    test_call_default,
    r#"
from dataclasses import dataclass
@dataclass
class A:
    x: int = int()
A()  # OK
    "#,
);

testcase!(
    test_override,
    r#"
import dataclasses
class A:
    pass
class B:
    def f(self, x: A) -> None:
        raise NotImplementedError()
@dataclasses.dataclass(frozen=True)
class C(B):
    def f(self, x: A) -> None:
        pass
    "#,
);

testcase!(
    test_initvar_parameter_types,
    r#"
from dataclasses import dataclass, field, InitVar

@dataclass
class InitVarTest:
    value: int = field(init=False)
    mode: InitVar[str]
    count: InitVar[int]

    def __post_init__(self, mode: str, count: int):
        if mode == "number":
            self.value = count * 10
        else:
            self.value = 0

# InitVar[str] should accept str arguments, not InitVar[str] arguments
InitVarTest("number", 5)  # OK
InitVarTest("text", 3)   # OK
    "#,
);

testcase!(
    test_initvar_multiple_type_arguments,
    r#"
from dataclasses import dataclass, InitVar

@dataclass
class C:
    x: InitVar[int, str]  # E: Expected 1 type argument for `InitVar`, got 2

@dataclass
class D:
    y: InitVar[int]  # OK
    "#,
);

testcase!(
    test_non_frozen_cannot_extend_frozen,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenBase:
    x: int

@dataclass
class MutableChild(FrozenBase):  # E: Cannot inherit non-frozen dataclass `MutableChild` from frozen dataclass `FrozenBase`
    y: str
    "#,
);

testcase!(
    test_frozen_cannot_extend_non_frozen,
    r#"
from dataclasses import dataclass

@dataclass
class MutableBase:
    x: int

@dataclass(frozen=True)
class FrozenChild(MutableBase):  # E: Cannot inherit frozen dataclass `FrozenChild` from non-frozen dataclass `MutableBase`
    y: str
    "#,
);

testcase!(
    test_frozen_can_extend_frozen,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenBase:
    x: int

@dataclass(frozen=True)
class FrozenChild(FrozenBase):  # OK
    y: str
    "#,
);

testcase!(
    test_non_frozen_can_extend_non_frozen,
    r#"
from dataclasses import dataclass

@dataclass
class MutableBase:
    x: int

@dataclass
class MutableChild(MutableBase):  # OK
    y: str
    "#,
);

testcase!(
    test_initvar_not_stored_as_attributes,
    r#"
from dataclasses import dataclass, field, InitVar
@dataclass
class InitVarTest:
    value: int = field(init=False)
    mode: InitVar[str]
    count: InitVar[int]
    def __post_init__(self, mode: str, count: int):
        if mode == "number":
            self.value = count * 10
        else:
            self.value = 0
instance = InitVarTest("number", 5)
# InitVar fields should not be accessible as instance attributes
instance.mode  # E: Object of class `InitVarTest` has no attribute `mode`
instance.count  # E: Object of class `InitVarTest` has no attribute `count`
# Regular fields should be accessible
instance.value  # OK
    "#,
);

testcase!(
    test_dataclass_kw_only,
    r#"
from dataclasses import dataclass

@dataclass(kw_only=False)
class SomeClass:
    x: int

SomeClass(x=1) # OK
SomeClass(1) # OK
    "#,
);

testcase!(
    test_dataclass_field_kw_only_override_class_true,
    r#"
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class SomeClass:
    x: int = field(kw_only=False)

SomeClass(x=1) # OK
SomeClass(1) # OK
    "#,
);

testcase!(
    test_alias,
    r#"
from dataclasses import dataclass

mutable = dataclass
@mutable
class Mut:
    x: int
m = Mut(x=0)
m.x = 42

frozen = dataclass(frozen=True)
@mutable(frozen=True)
class Froz1:
    x: int
@frozen
class Froz2:
    x: int
froz1 = Froz1(x=0)
froz1.x = 42  # E: frozen dataclass member
froz2 = Froz2(x=0)
froz2.x = 42  # E: frozen dataclass member
    "#,
);

fn dataclass_alias_env() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from dataclasses import dataclass
mutable = dataclass
frozen = dataclass(frozen=True)
    "#,
    )
}

testcase!(
    test_imported_alias,
    dataclass_alias_env(),
    r#"
import foo

@foo.mutable
class Mut:
    x: int
m = Mut(x=0)
m.x = 42

@foo.mutable(frozen=True)
class Froz1:
    x: int
@foo.frozen
class Froz2:
    x: int
froz1 = Froz1(x=0)
froz1.x = 42  # E: frozen dataclass member
froz2 = Froz2(x=0)
froz2.x = 42  # E: frozen dataclass member
    "#,
);

testcase!(
    test_field_ordering_valid_no_defaults,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int
    y: str
    z: float
C(1, "hello", 3.14)  # OK
    "#,
);

testcase!(
    test_field_ordering_valid_all_defaults,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int = 1
    y: str = "hello"
    z: float = 3.14
C()  # OK
C(x=2)  # OK
C(x=2, y="world", z=2.71)  # OK
    "#,
);

testcase!(
    test_field_ordering_valid_required_then_defaults,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int
    y: str
    z: float = 3.14
C(1, "hello")  # OK
C(1, "hello", 2.71)  # OK
    "#,
);

testcase!(
    test_post_init_defining_attrs,
    r#"
from dataclasses import dataclass
from typing import assert_type

@dataclass
class Magic:
    foo: int
    def __post_init__(self):
        self.bar: int = 1
magic = Magic(foo=1)
assert_type(magic.foo, int)
assert_type(magic.bar, int)
    "#,
);

testcase!(
    test_field_ordering_basic_violation,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    x: int = 1
    y: str  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_multiple_violations,
    r#"
from dataclasses import dataclass
@dataclass
class C:
    a: int = 1
    b: str  # E: Dataclass field `b` without a default may not follow dataclass field with a default
    c: int = 2
    d: float  # E: Dataclass field `d` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_with_field_function,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(default=1)
    y: str  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_with_empty_field_function,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(default=1)  # Has DEFAULT flag AND is initialized on class
    y: int = field()           # E: Dataclass field `y` without a default may not follow dataclass field with a default
    z: int                     # E: Dataclass field `z` without a default may not follow dataclass field with a default
C(y=2, z=3)  # OK - y is not considered to have a default
    "#,
);

testcase!(
    test_field_ordering_with_default_factory,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: list[int] = field(default_factory=list)
    y: str  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_kw_only_bypass,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = 1
    y: str = field(kw_only=True)  # OK - kw_only fields bypass ordering validation
    z: int = field(kw_only=True)  # OK - kw_only fields bypass ordering validation
C(1, y="hello", z=2)  # OK
    "#,
);

testcase!(
    test_field_ordering_kw_only_sentinel,
    r#"
from dataclasses import dataclass, KW_ONLY
@dataclass
class C:
    x: int = 1
    _: KW_ONLY
    y: str  # OK - fields after KW_ONLY marker are keyword-only
    z: int  # OK - fields after KW_ONLY marker are keyword-only
C(1, y="hello", z=2)  # OK
    "#,
);

testcase!(
    test_field_ordering_kw_only_global,
    r#"
from dataclasses import dataclass
@dataclass(kw_only=True)
class C:
    x: int = 1
    y: str  # OK - all fields are keyword-only when kw_only=True
C(x=1, y="hello")  # OK
    "#,
);

testcase!(
    test_field_ordering_init_false_bypass,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = 1
    y: str = field(init=False)  # OK - init=False fields bypass ordering validation
    z: int  # E: Dataclass field `z` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_mixed_bypass_flags,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    a: int
    b: str = "default"
    c: float = field(kw_only=True)  # OK - kw_only field
    d: int = field(init=False)      # OK - init=False field
    e: bool  # E: Dataclass field `e` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_inheritance_violation,
    r#"
from dataclasses import dataclass
@dataclass
class Base:
    x: int = 1

@dataclass
class Child(Base):
    y: str  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_inheritance_valid,
    r#"
from dataclasses import dataclass
@dataclass
class Base:
    x: int

@dataclass
class Child(Base):
    y: str = "default"  # OK
Child(1, y="hello")  # OK
    "#,
);

testcase!(
    test_field_ordering_multiple_inheritance,
    r#"
from dataclasses import dataclass
@dataclass
class Base1:
    x: int = 1

@dataclass
class Base2:
    y: str = "default"

@dataclass
class Child(Base1, Base2):
    z: float  # E: Dataclass field `z` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_inheritance_with_kw_only,
    r#"
from dataclasses import dataclass, field
@dataclass
class A:
    a: int

@dataclass
class B:
    b: str = "default"

@dataclass
class C(A, B):  # E: Dataclass field `a` without a default may not follow dataclass field with a default
    c: float = field(kw_only=True)  # OK - kw_only
    d: bool  # E: Dataclass field `d` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_inherited_conflict_not_repeated_on_subclass,
    r#"
from dataclasses import dataclass
@dataclass
class HasDefault:
    a: int = 0

@dataclass
class Origin(HasDefault):
    b: int  # E: Dataclass field `b` without a default may not follow dataclass field with a default

@dataclass
class Sub(Origin):
    pass
    "#,
);

testcase!(
    test_field_ordering_initvar_violation,
    r#"
from dataclasses import dataclass, InitVar
@dataclass
class C:
    x: int = 1
    init_var: InitVar[str]  # E: Dataclass field `init_var` without a default may not follow dataclass field with a default
    y: int  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_classvar_bypass,
    r#"
from typing import ClassVar
from dataclasses import dataclass
@dataclass
class C:
    x: int = 1
    class_var: ClassVar[str] = "ignored"  # OK - ClassVar fields bypass ordering validation
    y: int  # E: Dataclass field `y` without a default may not follow dataclass field with a default
    "#,
);

testcase!(
    test_field_ordering_kw_only_positional_override,
    r#"
from dataclasses import dataclass, field
@dataclass(kw_only=True)
class C:
    a: int = 1
    b: str = field(kw_only=False)                 # positional override, no default
    c: float = field(kw_only=False, default=3.14) # positional override, has default
    d: bool = field(kw_only=False)                # E: Dataclass field `d` without a default may not follow dataclass field with a default
C("hello", 3.14, a=1, d=True)
    "#,
);

testcase!(
    test_field_ordering_kw_only_mixed_overrides,
    r#"
from dataclasses import dataclass, field
@dataclass(kw_only=True)
class C:
    w: int
    x: str = field(kw_only=False)     # positional override
    y: float = field(kw_only=False)   # positional override
    z: bool
C("hello", 3.14, w=1, z=True)
    "#,
);

testcase!(
    test_field_ordering_kw_only_field_override,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    a: int
    b: str = field(kw_only=True)      # keyword-only field
    c: float = 3.14                   # positional field with default
    d: bool = field(kw_only=False)    # E: Dataclass field `d` without a default may not follow dataclass field with a default
C(1, 2.71, b="hello", d=True)
    "#,
);

testcase!(
    test_field_kw_only_unsupported,
    TestEnv::new_with_version(PythonVersion::new(3, 9, 0)),
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = 1
    y: int = field(kw_only=True)
    z: int # E: Dataclass field `z` without a default may not follow dataclass field with a default
C(5, y=2) # E: Missing argument `z` in function `C.__init__`
C(5, y=2, z=3)
    "#,
);

testcase!(
    test_field_ordering_kw_only_field_bypass,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = 1
    y: int = field(kw_only=True)  # OK - kw_only field bypasses ordering validation
    z: int # E: Dataclass field `z` without a default may not follow dataclass field with a default
C(5, y=2) # E: Missing argument `z` in function `C.__init__`
C(5, 1, y=2)
    "#,
);

testcase!(
    test_slots,
    r#"
from dataclasses import dataclass
from typing import assert_type, Literal
@dataclass
class NoSlots:
    x: int
@dataclass(slots=True)
class Slots:
    x: int
no_slots = NoSlots(x=0)
no_slots.__slots__ # E: no attribute `__slots__`
slots = Slots(x=0)
assert_type(slots.__slots__, tuple[Literal['x']])
    "#,
);

testcase!(
    test_match_args_no_init,
    r#"
from dataclasses import dataclass, field
from typing import assert_type
@dataclass
class C:
    x: int = field(init=False)
assert_type(C.__match_args__, tuple[()])
    "#,
);

testcase!(
    test_match_args_initvar,
    r#"
from dataclasses import dataclass, InitVar
from typing import assert_type, Literal
@dataclass
class C:
    x: InitVar[int]
assert_type(C.__match_args__, tuple[Literal['x']])
    "#,
);

// InitVars are passed positionally to `__post_init__`, in the order in which they're defined.
testcase!(
    test_post_init_validation,
    r#"
from dataclasses import dataclass, InitVar
@dataclass
class Good:
    x: int
    y: InitVar[str]
    z: InitVar[bytes]
    def __post_init__(self, y: str, z: bytes): ...
@dataclass
class Bad1:
    x: int
    y: InitVar[str]
    z: InitVar[bytes]
    def __post_init__(self, y: bytes, z: str): ...  # E: `__post_init__` type `(self: Bad1, y: bytes, z: str) -> None` is not assignable to expected type `(y: str, z: bytes) -> object` generated from the dataclass's `InitVar` fields
@dataclass
class Bad2:
    x: int
    y: InitVar[str]
    z: InitVar[bytes]
    def __post_init__(self, *, y: str, z: bytes): ...  # E: `__post_init__` type
    "#,
);

testcase!(
    test_descriptor,
    r#"
from dataclasses import dataclass
from typing import assert_type
class Desc:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
@dataclass
class C:
    x: Desc = Desc()  # E: Cannot set field `x` to data descriptor `Desc` with inconsistent types
c = C('')
assert_type(c.x, int)
c.x = 'cat'
c.x = 42  # E: `Literal[42]` is not assignable to parameter `value` with type `str` in function `Desc.__set__`
    "#,
);

testcase!(
    test_kwonly_mix,
    r#"
from dataclasses import dataclass, field
@dataclass(kw_only=True)
class C1:
    a: str = field(kw_only=False)
    b: int = 0
@dataclass
class C2(C1):
    c: float
C2('', 0.2, b=3)
    "#,
);

testcase!(
    test_kw_only_sentinel_inheritance,
    r#"
from dataclasses import dataclass, KW_ONLY

@dataclass
class Foo:
    _: KW_ONLY
    option: int | None = None

@dataclass
class Bar(Foo):
    arg: str

Bar("arg")
Bar(arg="arg")
    "#,
);

testcase!(
    test_assign_to_field_in_child,
    r#"
from dataclasses import dataclass

@dataclass
class Animal:
    name: str | None = None
    def speak(self) -> str: ...

@dataclass
class Dog(Animal):
    def speak(self) -> str:
        self.name = "dog"
        return "woof"

hdog = Dog(name="hdog")
    "#,
);

testcase!(
    test_fields_function,
    r#"
from typing import assert_type, Any
from dataclasses import dataclass, fields, Field

@dataclass
class Person:
    name: str
    age: int

# Test fields() on the class type
f1 = fields(Person)
assert_type(f1, tuple[Field[Any], ...])

# Test fields() on an instance
p = Person("Alice", 30)
f2 = fields(p)
assert_type(f2, tuple[Field[Any], ...])
    "#,
);

testcase!(
    test_final_field_no_modification,
    r#"
from typing import Final
from dataclasses import dataclass
@dataclass
class C:
    x: Final[int]
    y: Final[int] = 42

c = C(x=0)
c.x = 1  # E: Cannot set field `x`
c.y = 1  # E: Cannot set field `y`

C.x = 1  # E: Cannot set field `x`
C.y = 1  # E: Cannot set field `y`
"#,
);

testcase!(
    test_field_has_unknown_default,
    r#"
from dataclasses import dataclass, field
@dataclass
class C:
    x: int = field(default_factory=42) # E:
C()
    "#,
);

testcase!(
    test_non_data_descriptor_in_dataclass,
    r#"
from dataclasses import dataclass
from typing import assert_type, Self

# Non-data descriptors (only __get__, no __set__) in dataclasses are unsound:
# The dataclass __init__ writes to the instance dict, shadowing the class-level
# descriptor. This means the static type (from __get__) doesn't match the runtime
# type (the raw descriptor object in the instance dict).
class DescA:
    def __get__(self, obj, cls) -> int: ...
    # No __set__ - non-data descriptor

# If the result of `__get__` is `Self`, then the shadowing described above doesn't cause
# any static typing issues. Because this pattern does sometimes occur (e.g. Pytorch Device is a
# Self-returning descriptor), we allow it.
class DescB:
    def __get__(self, obj, cls) -> Self: ...
    # No __set__ - non-data descriptor, but __get__ returns Self

@dataclass
class C:
    x: DescA = DescA()  # E: Cannot set field `x` to non-data descriptor `DescA`\n  Hint: add a `__set__` method to make `DescA` a data descriptor
    y: DescB = DescB()

# Regardless of any errors, any descriptors assigned in the class body do have default values.
c = C()
    "#,
);

testcase!(
    test_non_data_descriptor_returns_own_class,
    r#"
from dataclasses import dataclass
from typing import assert_type

# A __get__ returning the descriptor's own class (not literal Self) is sound.
class Dev:
    def __get__(self, obj, cls) -> "Dev": ...

class Other: ...
class Bad:
    def __get__(self, obj, cls) -> Other: ...

@dataclass
class Base:
    x: Dev = Dev()

@dataclass
class Sub(Base):
    pass

@dataclass
class C:
    y: Bad = Bad()  # E: Cannot set field `y` to non-data descriptor `Bad`

assert_type(Base().x, Dev)
assert_type(Sub().x, Dev)
    "#,
);

testcase!(
    test_data_descriptor_in_dataclass,
    r#"
from dataclasses import dataclass
from typing import assert_type

# Data descriptors (have __set__) in dataclasses may work correctly because
# assignments go through the descriptor protocol rather than shadowing.
class DescA:
    def __get__(self, obj, cls) -> int: ...
    def __set__(self, obj, value: int) -> None: ...

# But if the `__get__` type does not match `__set__` then the default is
# incorrectly typed.
class DescB:
    def __get__(self, obj, cls) -> int: ...
    def __set__(self, obj, value: str) -> None: ...

@dataclass
class C:
    x: DescA = DescA()
    y: DescB = DescB()  # E: Cannot set field `y` to data descriptor `DescB` with inconsistent types\n  Return type `int` of `DescB.__get__` is not assignable to value type `str` of `DescB.__set__`

# The field has a default, and accepts the `__set__` type if provided.
c = C()
c = C(x=42, y='42')

# Reading should return the __get__ return type
assert_type(c.x, int)
assert_type(c.y, int)
    "#,
);

testcase!(
    test_dataclass_generic_descriptor_conformance,
    r#"
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, assert_type, overload

T = TypeVar("T")

class Desc2(Generic[T]):
    @overload
    def __get__(self, instance: None, owner: Any) -> list[T]: ...
    @overload
    def __get__(self, instance: object, owner: Any) -> T: ...
    def __get__(self, instance: object | None, owner: Any) -> list[T] | T: ...

@dataclass
class DC2:
    x: Desc2[int]
    y: Desc2[str]
    z: Desc2[str] = Desc2()  # E: Cannot set field `z` to non-data descriptor `Desc2`

assert_type(DC2.x, list[int])
assert_type(DC2.y, list[str])

dc2 = DC2(Desc2(), Desc2(), Desc2())
assert_type(dc2.x, int)
assert_type(dc2.y, str)
"#,
);

testcase!(
    test_dataclass_slots_undeclared_attr_conformance,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class DC2:
    x: int

    def __init__(self):
        self.x = 3
        # should error: y is not in slots
        self.y = 3  # E: not declared in `__slots__`

@dataclass(slots=False)
class DC3:
    x: int
    __slots__ = ("x",)

    def __init__(self):
        self.x = 3
        # should error: y is not in slots
        self.y = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_dataclass_protocol_dataclass_fields,
    r#"
from dataclasses import dataclass, Field
from typing import Any, ClassVar, Protocol

class P(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

@dataclass
class C(P):
    x: int

C(42)
"#,
);

// https://github.com/facebook/pyrefly/issues/2923
testcase!(
    test_dataclass_on_named_tuple,
    r#"
from dataclasses import dataclass
from typing import NamedTuple

class Coord(NamedTuple):
    x: int
    y: int

dataclass(Coord)  # E: Cannot apply `@dataclass` to NamedTuple `Coord`
"#,
);

// https://github.com/facebook/pyrefly/issues/2921
testcase!(
    test_dataclass_on_protocol,
    r#"
from dataclasses import dataclass
from typing import Protocol

class Printable(Protocol):
    def display(self) -> str: ...

dataclass(Printable)  # E: `@dataclass` cannot be applied to Protocol `Printable`
"#,
);

testcase!(
    test_dataclass_on_enum,
    r#"
from dataclasses import dataclass
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

dataclass(Color)  # E: Cannot apply `@dataclass` to Enum `Color`
"#,
);

// https://github.com/facebook/pyrefly/issues/2921
testcase!(
    test_dataclass_decorator_on_protocol,
    r#"
from dataclasses import dataclass
from typing import Protocol

@dataclass
class MyProto(Protocol):  # E: `@dataclass` cannot be applied to Protocol
    x: int
    def display(self) -> str: ...

@dataclass
class DC:
    x: int

class DC2(Protocol, DC):  # E: If `Protocol` is included as a base class, all other bases must be protocols
    y: int
"#,
);

// https://github.com/facebook/pyrefly/issues/2921
testcase!(
    test_dataclass_protocol_fields_in_subclass,
    r#"
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Base(Protocol):  # E: `@dataclass` cannot be applied to Protocol
    x: int

@dataclass
class Child(Base):
    y: str

Base(0)  # E: Cannot instantiate `Base` because it is a protocol
Child(x=0, y="ok")
Child(0, "ok")
Child("bad", "ok")  # E: Argument `Literal['bad']` is not assignable to parameter `x` with type `int` in function `Child.__init__`
"#,
);

// https://github.com/facebook/pyrefly/issues/3751
testcase!(
    test_dataclass_decorator_on_named_tuple,
    r#"
from dataclasses import dataclass
from typing import NamedTuple

@dataclass
class Foo(NamedTuple):  # E: Cannot apply `@dataclass` to NamedTuple
    x: int
"#,
);

// https://github.com/facebook/pyrefly/issues/2920
testcase!(
    test_frozen_dataclass_override_setattr_delattr,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class Immutable:
    value: int

    def __setattr__(self, name: str, val: object) -> None: ...  # E: Cannot override `__setattr__` in a frozen dataclass
    def __delattr__(self, name: str) -> None: ...  # E: Cannot override `__delattr__` in a frozen dataclass
"#,
);

// Subclass of a frozen dataclass: only `BadOverride` fires (it carries the
// richer "declared as final in parent class `Base`" message). Our
// `Cannot override __setattr__/__delattr__ in a frozen dataclass`
// diagnostic is suppressed here because it is scoped to the class that is
// itself decorated with `@dataclass(frozen=True)`.
testcase!(
    test_frozen_dataclass_subclass_override_setattr,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class Base:
    value: int

class Child(Base):
    def __setattr__(self, name: str, val: object) -> None: ...  # E: `__setattr__` is declared as final in parent class `Base`
    def __delattr__(self, name: str) -> None: ...  # E: `__delattr__` is declared as final in parent class `Base`
"#,
);

// Doubly-frozen: child is also `@dataclass(frozen=True)`. The parent already
// synthesizes `@final __setattr__`/`__delattr__`, so only `BadOverride` fires.
testcase!(
    test_frozen_dataclass_doubly_frozen_override_setattr,
    r#"
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenBase:
    value: int

@dataclass(frozen=True)
class FrozenChild(FrozenBase):
    def __setattr__(self, name: str, val: object) -> None: ...  # E: `__setattr__` is declared as final in parent class `FrozenBase`
    def __delattr__(self, name: str) -> None: ...  # E: `__delattr__` is declared as final in parent class `FrozenBase`
"#,
);

// Non-frozen dataclass should allow overriding __setattr__ and __delattr__
testcase!(
    test_non_frozen_dataclass_override_setattr_ok,
    r#"
from dataclasses import dataclass

@dataclass
class Mutable:
    value: int

    def __setattr__(self, name: str, val: object) -> None: ...
    def __delattr__(self, name: str) -> None: ...
"#,
);

testcase!(
    test_field_without_annotation,
    r#"
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class HasUnannotatedField:
    idx = field(default=1)  # E: `idx` is a dataclass field but has no type annotation
    another: int

@dataclass
class HasAnnotatedField:
    another: int
    idx: int = field(default=1)  # !E: type annotation

@dataclass
class HasClassVarField:
    x: ClassVar[int] = field(default=1)  # !E: type annotation

class NotADataclass:
    # Outside a dataclass, an unannotated `field()` is just an ordinary assignment.
    x = field(default=1)  # !E: type annotation

def user_defined_field() -> None:
    # A user-defined `field` is not a recognized field specifier, so an
    # unannotated assignment to it is fine even inside a dataclass.
    def field(default: int = 0) -> int:
        return default
    @dataclass
    class C:
        x = field(default=1)  # !E: type annotation
"#,
);

// Unlike attrs, a stdlib dataclass does NOT strip leading underscores from a private
// field's `__init__` parameter.
testcase!(
    test_dataclass_private_field_keeps_underscore,
    r#"
from dataclasses import dataclass
from typing import reveal_type

@dataclass
class C:
    _x: int

reveal_type(C.__init__)  # E: revealed type: (self: C, _x: int) -> None
"#,
);

// A dataclass field named 'self" must not collide with the implicit "self" parameter of the synthesized "__init__". cpython renames the instance param to "__dataclass_self__".
testcase!(
    test_dataclass_field_named_self,
    r#"
from dataclasses import dataclass
from typing import assert_type

@dataclass
class C:
    self: str

c = C(self="test")
assert_type(c.self, str)
"#,
);
