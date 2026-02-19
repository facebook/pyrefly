/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_typed_dict,
    r#"
from typing import TypedDict, Mapping
class Coord(TypedDict):
    x: int
    y: int
def foo(c: Coord) -> Mapping[str, object]:
    return c
def bar(c: Coord) -> Mapping[str, int]:
    return c  # E: is not assignable
def baz(c: Coord) -> Mapping[str, str]:
    return c  # E: is not assignable
"#,
);

testcase!(
    bug = "Our handling of ClassVar and methods is fishy, and our error messages are not clear",
    test_typed_dict_with_illegal_members,
    r#"
from typing import Any, TypedDict, ClassVar, assert_type
# Although classmethods, classvars, and static methods do actually
# work at runtime, type checkers seem to agree that these are not
# permissible in typed dicts.
class D(TypedDict):
    cv: ClassVar[int]  # E: `ClassVar` may not be used for TypedDict members
    x: str = "x"  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
    z = "z"  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
    def f(self) -> None:  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
        self.w = "w"  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
    @classmethod
    def g(cls) -> None:  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
        cls.u = "u"  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
    @staticmethod
    def h(self) -> None:  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
        ...
def foo(d: D):
    assert_type(d["cv"], int)
    assert_type(d["x"], str)
    assert_type(d["z"], Any)  # E: TypedDict `D` does not have key `z`
    assert_type(d["f"], Any)  # E: TypedDict `D` does not have
    assert_type(d["g"], Any)  # E: TypedDict `D` does not have
    assert_type(d["h"], Any)  # E: TypedDict `D` does not have
    assert_type(d["w"], Any)  # E: TypedDict `D` does not have
    assert_type(d["u"], Any)  # E: TypedDict `D` does not have
    assert_type(D.cv, int)
    assert_type(D.g, Any)
    assert_type(D.h, Any)
    "#,
);

testcase!(
    test_typed_dict_kwargs_type,
    r#"
from typing import assert_type, TypedDict, Unpack
class Coord(TypedDict):
    x: int
    y: int
def test1(**kwargs: Coord):
    assert_type(kwargs, dict[str, Coord])
def test2(**kwargs: Unpack[Coord]):
    assert_type(kwargs, Coord)
    "#,
);

testcase!(
    test_qualifiers,
    r#"
from typing import TypedDict, Required, NotRequired, ReadOnly, ClassVar, Final
class MyDict(TypedDict):
    v: ClassVar[int]  # E: `ClassVar` may not be used for TypedDict members
    w: Final[int]  # E: `Final` may not be used for TypedDict members
    x: NotRequired  # E: Expected a type argument for `NotRequired`
    y: Required  # E: Expected a type argument for `Required`
    z: ReadOnly  # E: Expected a type argument for `ReadOnly`
    "#,
);

testcase!(
    test_typed_dict_invalid_inheritance,
    r#"
from typing import TypedDict
class Coord(TypedDict, object):  # E: Typed dictionary definitions may only extend other typed dictionaries
    x: int
    y: int
    "#,
);

testcase!(
    test_typed_dict_literal,
    r#"
from typing import TypedDict, NotRequired
class Coord(TypedDict):
    x: int
    y: int
    z: NotRequired[int]

c1: Coord = {"x": 1, "y": 2}
c2: Coord = {"x": 1, "y": 2, "z": 3}
c3: Coord = {"x": 1, "y": 2, "a": 4}  # E: Key `a` is not defined in TypedDict `Coord`
c4: Coord = {"x": 1, "y": "foo"}  # E: `Literal['foo']` is not assignable to TypedDict key `y` with type `int`
c5: Coord = {"x": 1}  # E: Missing required key `y` for TypedDict `Coord`
c6: Coord = {"x": 1, **{"y": 2, **{"z": 3}}}
d: dict[str, int] = {}
c7: Coord = {"x": 1, **d}  # E: Unpacked `dict[str, int]` is not assignable to `Coord`

def foo(c: Coord) -> None:
    pass
foo({"x": 1, "y": 2})
    "#,
);

testcase!(
    test_typed_dict_callable,
    r#"
from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int
m = Movie(name='Blade Runner', year=1982)
    "#,
);

testcase!(
    test_typed_dict_read_only,
    r#"
from typing import TypedDict, ReadOnly
class Coord(TypedDict):
    x: int
    y: ReadOnly[int]
def foo(c: Coord) -> None:
    c["x"] = 1
    c["x"] = "foo"  # E: `Literal['foo']` is not assignable to TypedDict key `x` with type `int`
    c["y"] = 3  # E: Key `y` in TypedDict `Coord` is read-only
    c["z"] = 4  # E: TypedDict `Coord` does not have key `z`
    "#,
);

testcase!(
    test_setitem_union,
    r#"
from typing import Any, TypedDict
class C(TypedDict):
    x: int
def f(c: C | Any):
    c["x"] = 0
    "#,
);

testcase!(
    test_typed_dict_readonly_partial_update,
    r#"
from typing import Never, NotRequired, TypedDict, ReadOnly
from typing_extensions import ReadOnly

class A(TypedDict):
    x: ReadOnly[int]
    y: int

a1: A = {"x": 1, "y": 2}
a2: A = {"x": 3, "y": 4}
a1.update(a2) # E: No matching overload

class B(TypedDict):
    x: NotRequired[Never]
    y: ReadOnly[int]

def update_a(a: A, b: B) -> None:
    a.update(b)
    "#,
);

testcase!(
    test_update_with_readonly_key,
    r#"
from typing import ReadOnly, TypedDict
class A(TypedDict):
    x: ReadOnly[int]
a: A = {'x': 1}
a.update({'x': 2})  # E: No matching overload found for function `A.update`
    "#,
);

testcase!(
    test_typed_dict_readonly_kwargs_tuple_update,
    r#"
from typing import TypedDict, ReadOnly

class A(TypedDict):
    x: ReadOnly[int]
    y: int

def test(a: A) -> None:
    a.update([("x", 123), ("y", 456)])  # E: No matching overload found for function `A.update`
    a.update([("y", 456)])
    a.update(x=789, y=999)  # E: No matching overload found for function `A.update`
    a.update(y=999)
    "#,
);

testcase!(
    test_typed_dict_inheritance_field_qualifiers,
    r#"
from typing import NotRequired, ReadOnly, Required, TypedDict

class ParentRequired(TypedDict):
    x: int

class ChildOptional(ParentRequired):
    x: NotRequired[int]  # E: TypedDict field `x` in `ChildOptional` must remain required because parent TypedDict `ParentRequired` defines it as required

class ParentOptional(TypedDict, total=False):
    x: int

class ChildRequired(ParentOptional):
    x: Required[int]  # E: TypedDict field `x` in `ChildRequired` cannot be made required; parent TypedDict `ParentOptional` defines it as non-required

class ParentMutable(TypedDict):
    x: int

class ChildReadOnly(ParentMutable):
    x: ReadOnly[int]  # E: TypedDict field `x` in `ChildReadOnly` cannot be marked read-only; parent TypedDict `ParentMutable` defines it as mutable
"#,
);

testcase!(
    test_typed_dict_contextual,
    r#"
from typing import TypedDict
class MyDict(TypedDict, total=False):
    data: list[str | int]
def test():
    s: MyDict = {}
    s['data'] = []
    s['data'] = [42]
    s['data'] = [42, 'hello']
    s['data'] = ['hello']
    "#,
);

testcase!(
    test_typed_dict_metaclass,
    r#"
from enum import EnumMeta
from typing import TypedDict
class Coord(TypedDict, metaclass=EnumMeta):  # E: Typed dictionary definitions may not specify a metaclass
    x: int
    y: int
    "#,
);

testcase!(
    test_typed_dict_iterate,
    r#"
from typing import TypedDict, assert_type
class Coord(TypedDict):
    x: int
    y: int
def foo(c: Coord) -> None:
    for x in Coord:  # E: Type `type[Coord]` is not iterable
        pass
    for x in c:
        assert_type(x, str)
    "#,
);

testcase!(
    test_typed_dict_generic,
    r#"
from typing import TypedDict
class Coord[T](TypedDict):
    x: T
    y: T
def foo(c: Coord[int]):
    x: int = c["x"]
    y: str = c["y"]  # E: `int` is not assignable to `str`
    "#,
);

testcase!(
    test_typed_dict_access,
    r#"
from typing import TypedDict, Literal, assert_type
class Coord(TypedDict):
    x: int
    y: str
    z: bool
def foo(c: Coord, key: str, key2: Literal["x", "y"]):
    x: int = c["x"]
    x2: int = c.x  # E: Object of class `Coord` has no attribute `x`
    x3: int = c[key]  # E: Invalid key for TypedDict `Coord`, got `str`
    x4: int = c["aaaaaa"]  # E: TypedDict `Coord` does not have key `aaaaaa`
    assert_type(c[key2], int | str)
    "#,
);

testcase!(
    test_typed_dict_delete,
    r#"
from typing import TypedDict, ReadOnly, Required, LiteralString, NotRequired
class Coord(TypedDict, total=False):
    x: int
    y: ReadOnly[str]
    z: Required[bool]
class JustX(TypedDict, extra_items=int):
    x: NotRequired[int]
def foo(c: Coord, c2: Coord | JustX, c3: JustX, k: LiteralString):
    del c["x"]  # OK
    del c["y"]  # E: Key `y` in TypedDict `Coord` may not be deleted
    del c["z"]  # E: Key `z` in TypedDict `Coord` may not be deleted
    del c2["x"]  # OK
    del c3[k] # OK
    "#,
);

testcase!(
    test_typed_dict_functional,
    r#"
from typing import TypedDict, Required, NotRequired
Coord = TypedDict("Coord", { "x": Required[int], " illegal ": int, "y": NotRequired[int] })
c: Coord = {"x": 1, " illegal ": 2}
def test(c: Coord):
    x: int = c[" illegal "]
Invalid = TypedDict()  # E: Expected a callable, got `type[TypedDict]`
    "#,
);

testcase!(
    test_typed_dict_pop,
    r#"
from typing import TypedDict, NotRequired

class TD(TypedDict):
    x: NotRequired[int]

def f(td: TD):
    td.pop("x")
    "#,
);

testcase!(
    test_typed_dict_pop_2,
    r#"
from typing import TypedDict, NotRequired, assert_type, Any

class TDRequired(TypedDict):
    a: int
    b: str

class TDOptional(TypedDict):
    x: NotRequired[int]
    y: NotRequired[str]

class TDMixed(TypedDict):
    a: int
    x: NotRequired[int]

td_r: TDRequired = {"a": 10, "b": "hi"}
td_o: TDOptional = {"x": 42}
td_m: TDMixed = {"a": 1, "x": 99}

v1 = td_r.pop("a") # E:
assert_type(v1, object)

v2 = td_r.pop("a", 3.14) # E:
assert_type(v2, object)

v3 = td_o.pop("x")
assert_type(v3, int)

v4 = td_o.pop("x", -1)
assert_type(v4, int)

v5 = td_o.pop("x", "fallback")
assert_type(v5, int | str)

v6 = td_m.pop("a") # E:
assert_type(v6, Any)

v7 = td_m.pop("x")
assert_type(v7, int)

v8 = td_m.pop("x", 0)
assert_type(v8, int)

v9 = td_m.pop("x", "fallback")
assert_type(v9, int | str)

v10 = td_r.pop("abc", 123) # E:
assert_type(v10, object)

v11 = td_r.pop("abc", "default") # E:
assert_type(v11, object)
    "#,
);

testcase!(
    test_typed_dict_del,
    r#"
from typing import TypedDict, NotRequired

class TDRequired(TypedDict):
    a: int
    b: str

class TDOptional(TypedDict):
    x: NotRequired[int]
    y: NotRequired[str]

class TDMixed(TypedDict):
    a: int
    x: NotRequired[int]

td_r: TDRequired = {"a": 10, "b": "hi"}
td_o: TDOptional = {"x": 42}
td_m: TDMixed = {"a": 1, "x": 99}

del td_r["a"]  # E:
del td_o["x"]  # OK
del td_o["y"]  # OK
del td_m["x"]  # OK
del td_m["a"]  # E:
del td_r["nonexistent"]  # E:

# Delete optional key from TDOptional again
del td_o["x"]  # OK

del td_r["b"]  # E:

del td_o["unknown"]  # E:

key_var = "x"
del td_o[key_var]

unknown_key = "a"
del td_m[unknown_key]  # E:

td_r: TDRequired = {"a": 10, "b": "hi"}
td_o: TDOptional = {"x": 42}
td_m: TDMixed = {"a": 1, "x": 99}
    "#,
);

testcase!(
    bug = "These don't work because we special-case some operations on TypedDict rather than going through a magic method call",
    test_typed_dict_dunder_methods,
    r#"
from typing import assert_type, TypedDict, NotRequired

class TDOptional(TypedDict):
    x: NotRequired[int]
    y: NotRequired[str]

td_o: TDOptional = {"x": 42}
assert_type(td_o.__getitem__("x"), int)  # E: assert_type(object, int)
td_o.__setitem__("x", 42)  # E: no attribute `__setitem__`
td_o.__delitem__("x")  # E: not assignable to parameter `k` with type `Never`
    "#,
);

testcase!(
    test_typed_dict_dunder_or,
    r#"
from typing import TypedDict, assert_type

class TD1(TypedDict):
    a: int
    b: str

class TD2(TypedDict):
    c: float
    d: bool

class TD3(TypedDict, total=False):
    a: str
    f: int

class TD4(TypedDict):
    a: int

class TD5(TypedDict, total=False):
    a: int

td1 = TD1(a=1, b="x")
td2 = TD2(c=3.14, d=True)
td3 = TD3(a="str", f=5)
td4 = TD4(a=42)
td5 = TD5(a=100)

td12 = td1 | td2 # E: `|` is not supported between `TD1` and `TD2`
td14 = td1 | td4
assert_type(td14.get("a"), int)
td13 = td1 | td3 # E:   No matching overload found for function `_typeshed._type_checker_internals.TypedDictFallback.__or__`
td15 = td1 | td5 # E: No matching overload found for function `_typeshed._type_checker_internals.TypedDictFallback.__or__`
    "#,
);

testcase!(
    test_typed_dict_dunder_ior,
    r#"
from typing import TypedDict, assert_type

class TD(TypedDict):
    a: int | str
    b: int | str

x: TD = {"a": 1, "b": 2}

y: TD = {"a": "3", "b": "4"}

x |= y

assert_type((x["a"]), int | str)

x |= {} # E: Augmented assignment result `dict[str, object]` is not assignable to `TD`

x: TD = {"a": 1, "b": 2}
x.__ior__(y)
x.__ior__({}) # E:  Missing required key `a` for TypedDict `TD` # E: Missing required key `b` for TypedDict `TD`

    "#,
);

testcase!(
    test_typed_dict_dunder_ror,
    r#"
from typing import TypedDict, assert_type

class TD(TypedDict):
    a: int | str
    b: int | str

x: TD = {"a": 1, "b": 2}

result = {"a": "1", "b": "2", "c":4} | x
assert_type(result.get("c"), object | None)

result = x.__ror__({"a": "1", "b": "2", "c":4})
assert_type(result.get("c"), object | None)
    "#,
);

testcase!(
    test_typed_dict_subtype,
    r#"
from typing import TypedDict
class Coord(TypedDict):
    x: int
    y: int
class Coord3D(TypedDict):
    x: int
    y: int
    z: int
class Pair(TypedDict):
    x: object
    y: object

def foo(a: Coord, b: Coord3D, c: Pair):
    coord: Coord = b
    coord2: Coord3D = a  # E: `Coord` is not assignable to `Coord3D`
    coord3: Coord = c  # E: `Pair` is not assignable to `Coord`
    coord4: Pair = a  # E: `Coord` is not assignable to `Pair`
    "#,
);

testcase!(
    test_typed_dict_read_only_subtype,
    r#"
from typing import ReadOnly, TypedDict

class TD(TypedDict):
    a: ReadOnly[int]

class TD2(TypedDict):
    a: ReadOnly[object]

def foo(td: TD, td2: TD2) -> None:
    td: TD = td2  # E: `TD2` is not assignable to `TD`
    td2: TD2 = td
    "#,
);

testcase!(
    test_typed_dict_not_required,
    r#"
from typing import TypedDict, NotRequired
class Coord(TypedDict):
    x: int
    y: int
class CoordNotRequired(TypedDict):
    x: NotRequired[int]
    y: NotRequired[int]

def foo(a: Coord, b: CoordNotRequired):
    coord: Coord = b  # E: `CoordNotRequired` is not assignable to `Coord`
    coord2: CoordNotRequired = a  # E: `Coord` is not assignable to `CoordNotRequired`
    "#,
);

testcase!(
    test_typed_dict_totality,
    r#"
from typing import TypedDict, NotRequired

class CoordXY(TypedDict, total=True):
    x: int
    y: int
class CoordZ(TypedDict, total=False):
    z: int

class Coord(CoordZ):
    x: int
    y: int
class Coord2(CoordXY, total=False):
    z: int
class Coord3(TypedDict):
    x: int
    y: int
    z: NotRequired[int]
class Coord4(TypedDict, CoordXY, CoordZ):
    pass

def foo(a: Coord, b: Coord2, c: Coord3, d: Coord4):
    coord: Coord = b
    coord = c
    coord = d
    coord2: Coord2 = a
    coord2 = c
    coord2 = d
    coord3: Coord3 = a
    coord3 = b
    coord3 = d
    coord4: Coord4 = a
    coord4 = b
    coord4 = c
    "#,
);

testcase!(
    test_typed_dict_kwargs_expansion,
    r#"
from typing import TypedDict, NotRequired
class Coord(TypedDict):
    x: int
    y: int
    z: NotRequired[int]

def f1(x: int, y: int, z: int = 1): ...
def f2(x: int, y: int, **kwargs: str): ...
def f3(x: int, y: int): ...
def f4(x: int, y: int = 2, z: int = 3): ...
def f5(x: int, y: int, z: int): ...
def f6(x: int, y: int, **kwargs: str): ...

x: Coord = {"x": 1, "y": 2}
f1(**x)
f2(**x)  # E: Argument `int` is not assignable to parameter `z` with type `str` in function `f2`
f3(**x)  # E: Unexpected keyword argument `z`
f4(**x)
f5(**x)
f6(**x)  # E: Argument `int` is not assignable to parameter `z` with type `str` in function `f6`
f1(1, **x)  # E: Multiple values for argument `x`
    "#,
);

testcase!(
    test_typed_dict_kwargs_unpack,
    r#"
from typing import TypedDict, NotRequired, Unpack, assert_type
class Coord(TypedDict):
    x: int
    y: int
    z: NotRequired[int]

class Coord2(TypedDict):
    x: int
    y: int

class Coord3(TypedDict):
    x: int
    y: int
    z: int

class Coord4(TypedDict):
    w: int
    x: int
    y: int
    z: NotRequired[int]

class Coord5(TypedDict):
    y: int
    z: NotRequired[int]

def f(**kwargs: Unpack[Coord]):
    assert_type(kwargs["x"], int)

def g(x: Coord, x2: Coord2, x3: Coord3, x4: Coord4, x5: Coord5):
    f(**x)
    f(**x2)
    f(**x3)
    f(**x4)
    f(**x5)  # E: Missing argument `x`
f(x=1, y=2)
f(x=1, y=2, z=3)
f(x=1, y=2, z=3, a=4)  # E: Unexpected keyword argument `a`
f(x="", y=2)  # E: Argument `Literal['']` is not assignable to parameter `x` with type `int` in function `f`
    "#,
);

testcase!(
    test_typed_dict_read_only_variance,
    r#"
from typing import ReadOnly, TypedDict

class TD(TypedDict):
    a: int

class TD2(TypedDict):
    a: ReadOnly[int]

class TD3(TypedDict):
    a: bool

def foo(td2: TD2, td3: TD3) -> None:
    t1: TD2 = td3  # OK
    t2: TD = td2  # E: `TD2` is not assignable to `TD`
    t3: TD = td3  # E: `TD3` is not assignable to `TD`
    "#,
);

testcase!(
    test_inheritance,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
class B(A):
    y: str
B(x=0, y='1')  # OK
B(x=0, y=1)  # E: No matching overload found for function `B.__init__`
    "#,
);

testcase!(
    test_generic_instantiation,
    r#"
from typing import TypedDict, assert_type
class C[T](TypedDict):
     x: T
assert_type(C(x=0), C[int])
assert_type(C[str](x=""), C[str])
    "#,
);

testcase!(
    test_unpacked_typed_dict_assert_type,
    r#"
from typing import TypedDict, Unpack, assert_type
class Coord(TypedDict):
    x: int
    y: int
def foo(c: Coord, **kwargs: Unpack[Coord]):
    assert_type(c, Coord)
    assert_type(kwargs, Coord)
    "#,
);

testcase!(
    test_requireness_in_init,
    r#"
from typing import TypedDict, NotRequired
class D(TypedDict):
     x: int
     y: int = 5  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
     z: NotRequired[int]
# Default values are completely ignored in constructor behavior, so requiredness in `__init__` should be
# determined entirely by whether the field is required in the resulting dict.
D(x=5)  # E: No matching overload found for function `D.__init__`
    "#,
);

testcase!(
    test_cyclic_typed_dicts,
    r#"
from typing import TypedDict, assert_type
class TD0(TypedDict):
    x: int
    y: TD1
class TD1(TypedDict):
    x: int
    y: TD0
def foo(td0: TD0, td1: TD1) -> None:
    assert_type(td0, TD0)
    assert_type(td0['x'], int)
    assert_type(td0['y'], TD1)
    assert_type(td1, TD1)
    assert_type(td1['x'], int)
    assert_type(td1['y'], TD0)
    "#,
);

testcase!(
    test_typing_typeddict_functional,
    r#"
import typing
from typing import assert_type
X = typing.TypedDict('X', {'x': int})
assert_type(X, type[X])
    "#,
);

testcase!(
    test_assign_get_result,
    r#"
from typing import TypedDict
UserType1 = TypedDict("UserType1", {"name": str, "age": int}, total=False)
user1: UserType1 = {"name": "Bob", "age": 40}
name: str = user1.get("name", "n/a")
    "#,
);

testcase!(
    test_get,
    r#"
from typing import Any, Literal, TypedDict, assert_type
class C(TypedDict):
    x: int
    y: str
def f(c: C, k1: str, k2: int):
    assert_type(c.get("x"), int)
    assert_type(c.get(k1), object | None)
    assert_type(c.get(k1, 0), int | object)
    c.get(k2)  # E: No matching overload
    "#,
);

testcase!(
    test_get_not_required,
    r#"
from typing import assert_type, NotRequired, TypedDict
class C(TypedDict):
    x: NotRequired[int]
def f(c: C):
    assert_type(c.get("x"), int | None)
    "#,
);

// Clearing a TypedDict is not allowed, since doing so would remove keys it's expected to have.
testcase!(
    test_clear,
    r#"
from typing import TypedDict
class C(TypedDict):
    x: int
def f(c: C):
    c.clear()  # E: no attribute `clear`
    "#,
);

testcase!(
    test_update,
    r#"
from typing import TypedDict
class C(TypedDict):
    x: int
    y: int
class D(TypedDict):
    x: int
    y: int
class E(TypedDict):
    x: int
    y: str
class F(TypedDict):
    x: int
def f(c1: C, c2: C, c3: dict[str, int], d: D, e: E, f: F):
    c1.update(c2)
    c1.update(c3)  # E: No matching overload found for function `C.update`
    c1.update(d)
    c1.update(e)  # E: No matching overload found for function `C.update`
    # This is not ok because `F` could contain `y` with an incompatible type
    c1.update(f) # E: No matching overload found for function `C.update`
    c1.update({"x": 1, "y": 1})
    c1.update({"x": 1})
    c1.update({"z": 1})  # E: No matching overload found for function `C.update`
    c1.update([("x", 1), ("y", 2)])
    c1.update([("z", 3)]) # E: No matching overload found for function `C.update`
    c1.update(x=1, y=2)
    c1.update(z=1) # E: No matching overload found for function `C.update`
    "#,
);

testcase!(
    test_update_with_type_var,
    r#"
from typing import TypedDict

class X[T](TypedDict):
    a: T

def f(x: X[int], y: dict[str, int]):
    x.update(y)  # E: No matching overload found for function `X.update`
    x.update(x)
    x.update({"a": 1})
    x.update({"b": 1}) # E: No matching overload found for function `X.update`
    "#,
);

// Test an attribute that should be available on all TypedDict subclasses
testcase!(
    test_common_class_attr,
    r#"
from typing import TypedDict, assert_type
class C(TypedDict): ...
assert_type(C.__total__, bool)
    "#,
);

testcase!(
    bug = "Instances of TypedDicts are plain dicts at runtime and should not have a `__total__` attribute",
    test_class_attr_on_instance,
    r#"
from typing import TypedDict
class C(TypedDict): ...
def f(c: C):
    c.__total__  # This should be an error
    "#,
);

testcase!(
    test_setdefault,
    r#"
from typing import TypedDict, assert_type
class C(TypedDict):
    x: int
def f(c: C, s: str):
    assert_type(c.setdefault("x", 0), int)
    c.setdefault("x", 0.0)  # E: No matching overload
    c.setdefault("x")  # E: No matching overload
    c.setdefault(s, 0)  # E: No matching overload
    "#,
);

testcase!(
    test_setdefault_read_only,
    r#"
from typing import ReadOnly, TypedDict
class C(TypedDict):
    x: ReadOnly[int]
class D(TypedDict):
    x: int
    y: ReadOnly[str]
def f(c: C, d: D):
    c.setdefault("x", 0)  # E: `Literal['x']` is not assignable to parameter `k` with type `Never`
    d.setdefault("x", 0)
    d.setdefault("y", "oops")  # E: No matching overload
    "#,
);

testcase!(
    test_posonly_params,
    r#"
from typing import TypedDict
class C(TypedDict):
    x: int
def f(c: C):
    # Both of these methods take only positional arguments.
    c.get(key="x")  # E: No matching overload
    c.setdefault("x", default=0)  # E: No matching overload
    "#,
);

testcase!(
    test_kwonly_params_init,
    r#"
from typing import TypedDict
class C(TypedDict):
    x: int
C(0)  # E: No matching overload found for function `C.__init__`
    "#,
);

testcase!(
    test_attribute_named_self,
    r#"
from typing import TypedDict
class C(TypedDict):
    self: int
C(self=0)
    "#,
);

testcase!(
    test_required_and_notrequired_conflict,
    r#"
from typing import TypedDict, Required, NotRequired

class TD(TypedDict):
    x: Required[NotRequired[int]]  # E: Cannot combine `Required` and `NotRequired` for a TypedDict field
    y: NotRequired[Required[int]]  # E: Cannot combine `Required` and `NotRequired` for a TypedDict field
    z: NotRequired[NotRequired[int]]  # E: Duplicate qualifier `NotRequired`
    "#,
);

testcase!(
    test_typed_dict_isinstance_issubclass_not_allowed,
    r#"
from typing import TypedDict

class D(TypedDict):
    x: int

x = int
isinstance(x, D)  # E: TypedDict `D` not allowed as second argument to isinstance()
issubclass(x, D)  # E: TypedDict `D` not allowed as second argument to issubclass()
"#,
);

testcase!(
    test_typeddict_valid_base_class,
    r#"
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    "#,
);

testcase!(
    test_typeddict_invalid_annotations,
    r#"
from typing import TypedDict, TypeVar

T = TypeVar("T", bound=TypedDict)  # E: `TypedDict` is not allowed in this context

def test(x: TypedDict):  # E: `TypedDict` is not allowed in this context
    pass

x: TypedDict  # E: `TypedDict` is not allowed in this context
    "#,
);

testcase!(
    test_invalid_typed_dict_keywords,
    r#"
from typing import TypedDict

class TD1(TypedDict, total=True):  # OK
    x: int

class TD2(TypedDict, foo=1):  # E: TypedDict does not support keyword argument `foo`
    x: int

class TD3(TypedDict, bar="test", baz=False):  # E: TypedDict does not support keyword argument `bar`  # E: TypedDict does not support keyword argument `baz`
    x: int
"#,
);

testcase!(
    test_nonrequired_readonly,
    r#"
from typing import NotRequired, ReadOnly, Required, TypedDict
class A(TypedDict):
    x: NotRequired[ReadOnly[str]]
class B(TypedDict):
    x: Required[str]
b: B = {'x': ''}
a: A = b
    "#,
);

testcase!(
    test_value_error,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
# Test that the dict literal still matches `A` despite the error in `int(oops)`.
a: A | dict[int, str] = {'x': int(oops)}  # E: Could not find name `oops`
    "#,
);

testcase!(
    test_unpack_typed_dict_kwargs_validation,
    r#"
from typing import TypedDict, Unpack

class Coord(TypedDict):
    x: int
    y: int

class Person(TypedDict):
    name: str
    age: int

# Valid: Unpack[TypedDict] in **kwargs annotation
def valid_kwargs(**kwargs: Unpack[Coord]):
    pass

# Invalid: Unpack[TypedDict] in other contexts
def invalid_param(param: Unpack[Coord]):  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation
    pass

def invalid_return() -> Unpack[Coord]:  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation
    pass

x: Unpack[Coord] = ...  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation

# Invalid: Unpack with non-TypedDict in **kwargs
def invalid_unpack_int(**kwargs: Unpack[int]):  # E: `Unpack` in **kwargs annotation must be used only with a `TypedDict`
    pass

def invalid_unpack_dict(**kwargs: Unpack[dict[str, int]]):  # E: `Unpack` in **kwargs annotation must be used only with a `TypedDict`
    pass

# Valid: Regular TypedDict in **kwargs (without Unpack)
def regular_kwargs(**kwargs: Coord):
    pass
    "#,
);

testcase!(
    test_unpack_typed_dict_args_validation,
    r#"
from typing import TypedDict, Unpack

class Coord(TypedDict):
    x: int
    y: int

# Invalid: Unpack[TypedDict] in *args annotation
def invalid_args(*args: Unpack[Coord]):  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation
    pass

# Valid: Unpack with tuple in *args
def valid_args(*args: Unpack[tuple[int, str]]):
    pass
    "#,
);

testcase!(
    test_unpack_typed_dict_nested_validation,
    r#"
from typing import TypedDict, Unpack, Union

class Coord(TypedDict):
    x: int
    y: int

class Person(TypedDict):
    name: str
    age: int

# Invalid: Nested Unpack[TypedDict] outside **kwargs
def invalid_union(param: Union[int, Unpack[Coord]]):  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation
    pass

def invalid_list(param: list[Unpack[Coord]]):  # E: `Unpack` with a `TypedDict` is only allowed in a **kwargs annotation
    pass

# Valid: Unpack[TypedDict] only in **kwargs
def valid_kwargs(**kwargs: Unpack[Coord]):
    pass
    "#,
);

testcase!(
    test_items_as_key,
    r#"
from typing import assert_type, TypedDict
class TD(TypedDict):
    items: int
td: TD = {'items': 1}
assert_type(td['items'], int)
    "#,
);

testcase!(
    test_inheritance_consistency,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
class B(A):
    x: str  # E: `B.x` has type `str`, which is not consistent with `int` in `A.x`
    "#,
);

testcase!(
    test_closed_and_extra_items_kws,
    r#"
from typing import TypedDict
class TD1(TypedDict, closed=True):
    pass
class TD2(TypedDict, extra_items=str):
    pass
# Using `closed` and `extra_items` together is a runtime error.
class TD3(TypedDict, closed=True, extra_items=str):  # E: `closed` and `extra_items` cannot be used together
    pass
    "#,
);

testcase!(
    test_extra_items_requiredness,
    r#"
from typing import NotRequired, Required, TypedDict
class TD1(TypedDict, extra_items=NotRequired[int]):  # E: not allowed in this context
    pass
class TD2(TypedDict, extra_items=Required[int]):  # E: not allowed in this context
    pass
    "#,
);

testcase!(
    test_extra_items_readonly,
    r#"
from typing import ReadOnly, TypedDict
class TD(TypedDict, extra_items=ReadOnly[int]):
    pass
    "#,
);

testcase!(
    test_bad_extra_items,
    r#"
from typing import TypedDict
class TD(TypedDict, extra_items=False):  # E: Expected `extra_items` to be a type form, got instance of `Literal[False]`
    pass
    "#,
);

testcase!(
    test_construct_with_extra_items,
    r#"
from typing import TypedDict
class Movie(TypedDict, extra_items=int):
    name: str
good_movie: Movie = {'name': 'Toy Story', 'year': 1995}
bad_movie: Movie = {'name': 'Toy Story', 'studio': 'Pixar'}  # E: `Literal['Pixar']` is not assignable to TypedDict key with type `int`
    "#,
);

testcase!(
    test_kwargs_with_extra_items,
    r#"
from typing import TypedDict
class Movie(TypedDict, extra_items=int):
    name: str
good_movie = Movie(name='Toy Story', year=1995)
bad_movie = Movie(name='Toy Story', studio='Pixar')  # E: No matching overload found for function `Movie.__init__`
    "#,
);

testcase!(
    test_validate_bool_keyword,
    r#"
from typing import TypedDict
class Ok(TypedDict, total=True, closed=False):
    pass
def f() -> bool: ...
class Bad1(TypedDict, total=f()):  # E: Expected literal True or False for keyword `total`, got instance of `bool`
    pass
class Bad2(TypedDict, closed=f()):  # E: Expected literal True or False
    pass
    "#,
);

testcase!(
    test_closed_items_and_values,
    r#"
from typing import assert_type, TypedDict
class TD(TypedDict, closed=True):
    x: int
def f(td: TD):
    assert_type(list(td.items()), list[tuple[str, int]])
    assert_type(list(td.values()), list[int])
    "#,
);

testcase!(
    test_extra_items_and_values,
    r#"
from typing import assert_type, TypedDict
class TD(TypedDict, extra_items=int):
    x: str
def f(td: TD):
    assert_type(list(td.items()), list[tuple[str, int | str]])
    assert_type(list(td.values()), list[int | str])
    "#,
);

testcase!(
    test_cannot_unclose,
    r#"
from typing import TypedDict

class ClosedParent(TypedDict, closed=True):
    pass
class BadOpenChild1(ClosedParent, closed=False):  # E: Non-closed TypedDict cannot inherit from closed TypedDict `ClosedParent`
    pass

class ExtraItemsParent(TypedDict, extra_items=int):
    pass
class BadOpenChild2(ExtraItemsParent, closed=False):  # E: Non-closed TypedDict cannot inherit from TypedDict `ExtraItemsParent` with extra items
    pass
    "#,
);

testcase!(
    test_inherit_closed,
    r#"
from typing import assert_type, TypedDict
class Parent(TypedDict, closed=True):
    x: int
class Child(Parent):
    pass
def f(child: Child):
    assert_type(list(child.values()), list[int])
    "#,
);

testcase!(
    test_inherit_extra_items,
    r#"
from typing import TypedDict
class Parent(TypedDict, extra_items=int):
    x: str
class Child(Parent):
    pass
Child(x='ok', y=42)
    "#,
);

testcase!(
    test_can_close_if_readonly_extra,
    r#"
from typing import ReadOnly, TypedDict
class Parent1(TypedDict, extra_items=int):
    pass
class BadChild(Parent1, closed=True):  # E: Closed TypedDict cannot inherit from TypedDict `Parent1` with non-read-only extra items
    pass
class Parent2(TypedDict, extra_items=ReadOnly[int]):
    pass
class GoodChild1(Parent2, closed=True):
    pass
# A closed=False TypedDict (the default) is considered to have extra items of type `ReadOnly[object]`.
class Parent3(TypedDict):
    pass
class GoodChild2(Parent3, closed=True):
    pass
class Parent4(TypedDict, closed=False):
    pass
class GoodChild3(Parent4, closed=True):
    pass
    "#,
);

testcase!(
    test_change_extra_items,
    r#"
from typing import ReadOnly, TypedDict
class Parent1(TypedDict, extra_items=int):
    pass
class BadChild(Parent1, extra_items=bool):  # E: Cannot change the non-read-only extra items type of TypedDict `Parent1`
    pass
class Parent2(TypedDict, extra_items=ReadOnly[int]):
    pass
class GoodChild1(Parent2, extra_items=bool):  # ok because Parent2's extra_items is read-only
    pass
class Parent3(TypedDict):
    pass
class GoodChild2(Parent3, extra_items=bool):  # ok because Parent3 has extra items of type `ReadOnly[object]` by default
    pass
    "#,
);

testcase!(
    bug = "You shouldn't be able to add items to a closed TypedDict",
    test_no_add_items_if_closed,
    r#"
from typing import TypedDict
class Parent(TypedDict, closed=True):
    x: int
class Child(Parent):
    y: str  # E: Cannot extend closed TypedDict `Parent` with extra item `y`
    "#,
);

testcase!(
    test_add_items_with_readonly_extra_items,
    r#"
from typing import NotRequired, ReadOnly, Required, TypedDict
class Parent(TypedDict, extra_items=ReadOnly[int]):
    pass
class GoodChild(Parent):
    x: int
    y: Required[int]
    z: NotRequired[bool]
class BadChild(Parent):
    x: str  # E: `str` is not assignable to `extra_items` type `int` of TypedDict `Parent`
    "#,
);

testcase!(
    test_add_items_with_readwrite_extra_items,
    r#"
from typing import NotRequired, Required, TypedDict
class Parent(TypedDict, extra_items=int):
    pass
class GoodChild(Parent):
    x: NotRequired[int]
class BadChild1(Parent):
    x: Required[int]  # E: Cannot add required field `x`
class BadChild2(Parent):
    x: NotRequired[bool]  # E: `bool` is not consistent with `extra_items` type `int`
    "#,
);

testcase!(
    test_extra_items_assignability,
    r#"
from typing import ReadOnly, TypedDict
class A(TypedDict, extra_items=ReadOnly[int]):
    pass
class B(TypedDict, extra_items=bool):
    pass
class C(TypedDict):
    pass
a1: A = B()
# Not allowed because `C` implicitly has extra_items `ReadOnly[object]`, and `object` is not
# assignable to `int`.
a2: A = C()  # E: `C` is not assignable to `A`
    "#,
);

testcase!(
    test_extra_items_field_assignability,
    r#"
from typing import NotRequired, ReadOnly, TypedDict
class TDReadWrite(TypedDict, extra_items=int):
    pass
class TDReadOnly(TypedDict, extra_items=ReadOnly[int]):
    pass
class A(TypedDict, extra_items=int):
    x: NotRequired[int]
class B(TypedDict, extra_items=int):
    x: int
class C(TypedDict, extra_items=int):
    x: NotRequired[bool]

td1: TDReadWrite = A(x=0)
# not ok because `x` is required
td2: TDReadWrite = B(x=0)  # E: `B` is not assignable to `TDReadWrite`
# not ok because `bool` is not consistent with `int`
td3: TDReadWrite = C(x=True)  # E: `C` is not assignable to `TDReadWrite`

td4: TDReadOnly = A(x=0)
td5: TDReadOnly = B(x=0)
td6: TDReadOnly = C(x=True)
    "#,
);

testcase!(
    test_use_extra_items_for_readonly_missing_field,
    r#"
from typing import NotRequired, ReadOnly, TypedDict
class A(TypedDict):
    x: NotRequired[ReadOnly[int]]
class B(TypedDict, extra_items=bool):
    pass
class C(TypedDict, extra_items=str):
    pass
a1: A = B()
# Not ok because `str` is not assignable to `int`
a2: A = C()  # E: `C` is not assignable to `A`
    "#,
);

testcase!(
    test_use_extra_items_for_readwrite_missing_field,
    r#"
from typing import NotRequired, TypedDict
class A(TypedDict):
    x: NotRequired[int]
class B(TypedDict, extra_items=int):
    pass
class C(TypedDict, extra_items=bool):
    pass
a1: A = B()
# Not ok because `bool` is not consistent with `int`
a2: A = C()  # E: `C` is not assignable to `A`
    "#,
);

testcase!(
    test_update_with_extra_items,
    r#"
from typing import TypedDict
class A(TypedDict, extra_items=str):
    pass
class B(TypedDict, extra_items=str):
    pass
class C(TypedDict, extra_items=int):
    pass
def f(a: A, b: B, c: C):
    a.update(b)
    a.update(c)  # E: No matching overload
    a.update(x='ok')
    a.update([('x', 'ok')])
    "#,
);

testcase!(
    test_use_extra_items_for_update_missing_item,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
class B(TypedDict, extra_items=int):
    pass
class C(TypedDict, extra_items=str):
    pass
def f(a: A, b: B, c: C):
    a.update(b)
    a.update(c)  # E: No matching overload
    "#,
);

testcase!(
    test_functional_form_unexpected_keyword,
    r#"
from typing import TypedDict
X = TypedDict('X', {}, nonsense=True)  # E: Unrecognized keyword argument `nonsense` in typed dictionary definition
def f(kwargs):
    Y = TypedDict('Y', {}, **kwargs)  # E: Unpacking is not supported in typed dictionary definition
    "#,
);

testcase!(
    test_functional_extra_items,
    r#"
from typing import TypedDict
X = TypedDict('X', {}, extra_items=int)
x: X = {'x': 1}
y: X = {'y': 'oops'}  # E: `Literal['oops']` is not assignable to TypedDict key with type `int`
    "#,
);

testcase!(
    test_functional_closed,
    r#"
from typing import assert_type, TypedDict
X = TypedDict('X', {'x': int}, closed=True)
def f(x: X):
    assert_type(list(x.values()), list[int])
    "#,
);

testcase!(
    test_getitem_extra_items,
    r#"
from typing import assert_type, TypedDict
class A(TypedDict, extra_items=bool):
    a: str
def f(a: A, k: str):
    assert_type(a['x'], bool)
    assert_type(a[k], str | bool)
    "#,
);

testcase!(
    test_get_closed,
    r#"
from typing import assert_type, TypedDict
class A(TypedDict, closed=True):
    x: str
def f(a: A, k: str):
    assert_type(a[k], str)
    "#,
);

testcase!(
    test_typed_dict_kwargs_overlap_pos_arg,
    r#"
from typing import *
from functools import cache

class Kwargs(TypedDict):
    x: int

# this is OK because the first x is positional-only
def test(x: int, /, **kwargs: Unpack[Kwargs]): ...

# this should not be OK
def test(x: int, **kwargs: Unpack[Kwargs]): ... # E:  TypedDict key 'x' in **kwargs overlaps with parameter 'x'
@cache
def test(x: int, **kwargs: Unpack[Kwargs]): ... # E:  TypedDict key 'x' in **kwargs overlaps with parameter 'x'
    "#,
);

testcase!(
    test_set_extra_item,
    r#"
from typing import NotRequired, TypedDict

class A(TypedDict, extra_items=bool):
    a: NotRequired[bool]
def f(a: A, k: str):
    a['x'] = True
    a['y'] = 'oops'  # E: `Literal['oops']` is not assignable to TypedDict key `y` with type `bool`
    # This is allowed because `A` is assignable to `dict[str, bool]`.
    a[k] = False

class B(TypedDict, extra_items=bool):
    a: NotRequired[str]
def f(b: B, k: str):
    # This is not allowed because `B` is not assignable to a `dict` type: the `extra_items` type
    # `bool` is not consistent with type `str` of field `a`.
    b[k] = False  # E: Cannot set item
    "#,
);

testcase!(
    test_set_closed,
    r#"
from typing import TypedDict
class A(TypedDict, closed=True):
    x: str
def f(a: A, k: str):
    # This is not allowed because a closed TypedDict is not assignable to a `dict` type.
    a[k] = 'oops'  # E: Cannot set item
    "#,
);

testcase!(
    test_del_extra_items,
    r#"
from typing import NotRequired, TypedDict

class A(TypedDict, closed=True):
    x: NotRequired[str]
def f(a: A, k: str):
    del a['x']
    del a['y']  # E: `A` does not have key `y`
    # This is not allowed because a closed TypedDict is not assignable to a `dict` type.
    del a[k]  # E: Cannot delete item

class B(TypedDict, extra_items=int):
    x: NotRequired[str]
def f(b: B, k: str):
    del b['x']
    del b['y']
    # This is not allowed because `B` is not assignable to a `dict` type: `extra_items` type `int`
    # is not consistent with type `str` of field `x`.
    del b[k]  # E: Cannot delete item

class C(TypedDict, extra_items=int):
    x: NotRequired[int]
def f(c: C, k: str):
    del c['x']
    del c['y']
    del c[k]
    "#,
);

testcase!(
    test_unpack_with_extra_items,
    r#"
from typing import TypedDict, Unpack
class A(TypedDict, extra_items=int):
    x: str
def f(**kwargs: Unpack[A]):
    pass
f(x='ok', y=42)
f(x='no', y='oops')  # E: `Literal['oops']` is not assignable to parameter `**kwargs` with type `int`
    "#,
);

testcase!(
    test_mapping_assignability,
    r#"
from typing import Mapping, TypedDict
class A(TypedDict, extra_items=int):
    x: str
m1: Mapping[str, str | int] = A(x='')
m2: Mapping[str, int] = A(x='')  # E: `A` is not assignable to `Mapping[str, int]`
    "#,
);

testcase!(
    test_typed_dict_contains_itself,
    r#"
from typing import Mapping, TypedDict
class A(TypedDict, extra_items=int):
    x: "A"
def f(m: Mapping[str, A | int]):
    pass
def g(a: A):
    f(a)
    "#,
);

testcase!(
    test_dict_assignability,
    r#"
from typing import NotRequired, TypedDict
class A(TypedDict, extra_items=bool):
    x: NotRequired[bool]
d1: dict[str, bool] = A(x=True)
d2: dict[str, int] = A(x=False)  # E: `A` is not assignable to `dict[str, int]`
    "#,
);

testcase!(
    test_pop_readonly,
    r#"
from typing import NotRequired, ReadOnly, TypedDict
class A(TypedDict):
    x: NotRequired[ReadOnly[int]]
def f(a: A):
    a.pop('x')  # E: `Literal['x']` is not assignable to parameter `k` with type `Never`
    "#,
);

testcase!(
    test_pop_extra_items,
    r#"
from typing import assert_type, NotRequired, TypedDict
class A(TypedDict, extra_items=int):
    x: NotRequired[str]
def f(a: A, k: str):
    assert_type(a.pop('x'), str)
    assert_type(a.pop(k), str | int)
    assert_type(a.pop(k, b'default'), str | int | bytes)
    "#,
);

testcase!(
    test_setdefault_extra_items,
    r#"
from typing import assert_type, TypedDict
class A(TypedDict, extra_items=int):
    x: str
def f(a: A, k: str):
    assert_type(a.setdefault('x', ''), str)
    assert_type(a.setdefault(k, 0), int | str)
    "#,
);

testcase!(
    test_get_extra_items,
    r#"
from typing import assert_type, TypedDict
class A(TypedDict, extra_items=int):
    x: str
def f(a: A, k: str):
    assert_type(a.get(k), str | int | None)
    assert_type(a.get(k, b'hello world'), str | int | bytes)
    "#,
);

testcase!(
    test_remove_arbitrary_items,
    r#"
from typing import assert_type, NotRequired, TypedDict
class A(TypedDict, extra_items=int):
    x: NotRequired[str]
class B(TypedDict, extra_items=int):
    x: str
def f(a: A, b: B):
    assert_type(a.popitem(), tuple[str, int | str])
    a.clear()
    b.popitem()  # E: `B` has no attribute `popitem`
    b.clear()  # E: `B` has no attribute `clear`
    "#,
);

testcase!(
    test_builtins_dict_call,
    r#"
from typing import TypedDict
class A(TypedDict):
    x: int
a1: A = dict(x=1)
a2: A = dict(**{'x': 1})
a3: A = dict(x='oops')  # E: `Literal['oops']` is not assignable to TypedDict key `x`
a4: A = dict(**{'x': 'oops'})  # E: `Literal['oops']` is not assignable to TypedDict key `x`
    "#,
);

testcase!(
    test_annotated_typeddict,
    r#"
from typing import TypedDict

class MyDict(TypedDict):
    x: int
    y: str

fieldsets: tuple[tuple[str, MyDict], ...] | None = (
    ("A", {"x": 1, "y": "2"}),
)
    "#,
);

testcase!(
    test_type_of_typeddict_dunder_bool,
    r#"
from typing import TypedDict
class D1(TypedDict):
    a: int
    b: str
def func(d: type[D1] | None) -> int:
    if d:
        return 1
    return 2
    "#,
);

testcase!(
    test_unpack_inherited_typeddict,
    r#"
import typing_extensions as te

class InheritFromMe(te.TypedDict):
    foo: bool

class TestBadUnpackingError(InheritFromMe):
    bar: bool

unpack_this: InheritFromMe = {"foo": True}
test1: TestBadUnpackingError = {"bar": True, **unpack_this}
test2: TestBadUnpackingError = {"bar": True, "foo": True}
    "#,
);

testcase!(
    test_unexpected_item_in_unpacking,
    r#"
from typing import ReadOnly, TypedDict
class A(TypedDict):
    x: int
    y: str
class B1(TypedDict):
    x: int
class B2(TypedDict, extra_items=int):
    x: int
class B3(TypedDict, extra_items=ReadOnly[str]):
    x: int
class B4(TypedDict, extra_items=str):
    x: int
a: A = {'x': 0, 'y': ''}
# not ok because B1 does not have key `y`
b1: B1 = {'x': 0, **a}  # E: `A` is not assignable to `B1`
# not ok because extra items in B2 must have type `int`, not `str`
b2: B2 = {'x': 0, **a}  # E: `A` is not assignable to `B2`
# not ok because extra items in B2 are read-only
b3: B3 = {'x': 0, **a}  # E: `A` is not assignable to `B3`
# ok, `y` in A and extra items in B2 both have type `str`
b4: B4 = {'x': 0, **a}
    "#,
);

testcase!(
    test_open_unpacking,
    TestEnv::new().enable_open_unpacking_error(),
    r#"
from typing import TypedDict

class Open(TypedDict):
    x: int

class Closed(TypedDict, closed=True):
    x: int

class Extra(TypedDict, extra_items=str):
    x: int

class ExtraWrongType(TypedDict, extra_items=int):
    x: int

class Target(TypedDict):
    x: int
    y: str

open: Open = {'x': 0}
closed: Closed = {'x': 0}
extra: Extra = {'x': 0}
extra_wrong_type: ExtraWrongType = {'x': 0}

# Not ok, we could be unpacking a subclass of `open` with `y` with an unknown type
t1: Target = {'y': '', **open}  # E: open TypedDict with unknown extra items
# Ok, `closed` has no unknown items
t2: Target = {'y': '', **closed}
# Ok, `extra` has only extra items of the right type
t3: Target = {'y': '', **extra}
# Not ok, the extra items have the wrong type
t4: Target = {'y': '', **extra_wrong_type}  # E: `ExtraWrongType` is not assignable to `Target`
    "#,
);

testcase!(
    test_open_unpacking_closed_and_extra_items,
    TestEnv::new().enable_open_unpacking_error(),
    r#"
from typing import TypedDict

class Open(TypedDict):
    x: int

class ClosedTarget(TypedDict, closed=True):
    x: int

class ExtraItemsTarget(TypedDict, extra_items=int):
    x: int

open: Open = {'x': 0}

# not ok: `open` could contain arbitrary extra items when `t1` is supposed to be closed
t1: ClosedTarget = {**open}  # E: open TypedDict with unknown extra items
# not ok: `open` could contain extra items of the wrong type
t2: ExtraItemsTarget = {**open}  # E: open TypedDict with unknown extra items
    "#,
);

testcase!(
    test_union_as_key,
    r#"
from typing import TypedDict, Literal
class Foo(TypedDict):
    bar: int
    baz: int
def f(foo: Foo, k: Literal["bar", "baz"]):
    print(foo[k])
    foo[k] = 2
    "#,
);

testcase!(
    test_recursive_functional_typeddict,
    r#"
from typing import NamedTuple, TypedDict, Optional
ListNode = TypedDict('ListNode', {
    'value': int,
    'next': Optional['ListNode'],
})
"#,
);

testcase!(
    test_defines_init,
    r#"
from typing import TypedDict, Any
def any() -> Any: ...
class D(TypedDict):
    x: int
    __init__ = any()  # E: TypedDict members must be declared in the form `field: Annotation` with no assignment
D(x=5)
"#,
);

testcase!(
    test_typed_dict_missing_key_via_subscript,
    r#"
from typing import TypedDict
class TD(TypedDict):
    foo: int

td: TD = {"foo": 1}
td["fo"]  # E: TypedDict `TD` does not have key `fo`\n  Did you mean `foo`?
"#,
);

testcase!(
    test_notrequired_with_extra_items_as_kwargs,
    r#"
from typing import TypedDict, NotRequired

class TD(TypedDict, extra_items=str):
    field1: NotRequired[str]
    field2: NotRequired[str]

class TD2(TypedDict, extra_items=str):
    field1: NotRequired[str]

class TD3(TypedDict):
    field1: NotRequired[str]
    field2: NotRequired[str]

def fun(field1: str, field2: str):
    pass

def test(x: TD, y: TD2, z: TD3):
    fun(**x)
    fun(**y)  # E: Missing argument `field2` in function `fun`
    fun(**z)
"#,
);

testcase!(
    test_extra_items_literalstring_key,
    r#"
from typing import TypedDict, LiteralString

class TD(TypedDict, extra_items=str):
    ...
def test(x: TD, k1: LiteralString, k2: str):
    x[k1]
    x[k2]
"#,
);

testcase!(
    test_anonymous_typed_dict_usage,
    r#"
from typing import TypedDict, Unpack, Mapping

class TD(TypedDict):
    x: int
    y: str
d = { "x": 1, "y": "2" }
d = { "x": 2, "y": "3" }

def test(**kwargs: Unpack[TD]): pass
test(**d)
test(**{ "x": 1, "y": "2" })

def test2(**kwargs: str | int): pass
test2(**d)
test2(**{ "x": 1, "y": "2" })

def test3(x: int, y: str): pass
test3(**d)
test3(**{ "x": 1, "y": "2" })

d2: dict[str, str | int] = { "x": 1, "y": "2" }
d2 = d
m: Mapping[str, str | int] = { "x": 1, "y": "2" }
m = d
"#,
);

testcase!(
    test_anonymous_typed_dict_return,
    r#"
def foo():
    return {
        "a": "b",
        "c": {
            "d": "e"
        }
    }
foo()
x: str = foo()["a"]
"#,
);

testcase!(
    test_typed_dict_none_var_pinning,
    r#"
from typing import assert_type
x = { "foo": None }
x["foo"] = 1
# currently extra_items does not use the var, but we could
assert_type(x["bar"], None)
"#,
);

testcase!(
    test_typed_dict_constructor_with_dict_literal,
    r#"
from typing import TypedDict, NotRequired

class TD(TypedDict):
    x: int
    y: NotRequired[str]

x: TD = TD({"x": 1, "y": "2"})
x = TD({"x": 1}, y="2")
x = TD({"x": 1})
"#,
);

testcase!(
    test_generic_typed_dict_inheritance_with_unpack,
    r#"
from typing import TypedDict, Type, Generic
from typing_extensions import TypeVar, Unpack

_T = TypeVar("_T", default=str)

class Base(TypedDict, Generic[_T], total=False):
    default: _T | None

T = TypeVar('T')

class Child(Base[T], total=False):
    other: Type[T]

def test(**kwargs: Unpack[Child[int]]) -> None:
    pass

# Should accept default=5 because Child[int] should have default: int | None
test(other=int, default=5)
test(other=int, default="") # E: Argument `Literal['']` is not assignable to parameter `default` with type `int | None`
"#,
);

testcase!(
    test_typed_dict_contains_narrowing,
    r#"
from typing import TypedDict, Literal, reveal_type

class AClient: ...
class BClient: ...
class GenericClient: ...

class Clients(TypedDict):
    a: AClient
    b: BClient

def test_in(clients: Clients, name: str):
    if name in clients:
        reveal_type(name)  # E: revealed type: Literal['a', 'b']
        client = clients[name]
    else:
        client = GenericClient()
    return client

def test_not_in(clients: Clients, name: str):
    if name not in clients:
        reveal_type(name)  # E: revealed type: str
        return GenericClient()
    # name is narrowed in the else branch
    reveal_type(name)  # E: revealed type: Literal['a', 'b']
    client = clients[name]

def test_literal_union_in(clients: Clients, name: Literal['a', 'b', 'c']):
    # Test narrowing a literal union with 'in'
    if name in clients:
        reveal_type(name)  # E: revealed type: Literal['a', 'b']
        client = clients[name]
    else:
        # Only 'c' remains outside the TypedDict
        reveal_type(name)  # E: revealed type: Literal['c']
        client = GenericClient()
    return client

def test_literal_union_not_in(clients: Clients, name: Literal['a', 'b', 'c']):
    # Test narrowing a literal union with 'not in'
    if name not in clients:
        # Only 'c' is not in the TypedDict
        reveal_type(name)  # E: revealed type: Literal['c']
        return GenericClient()
    # 'a' and 'b' remain
    reveal_type(name)  # E: revealed type: Literal['a', 'b']
    client = clients[name]
    return client
"#,
);

testcase!(
    test_typed_dict_contains_narrowing_inheritance,
    r#"
from typing import TypedDict, Literal, reveal_type

class Base(TypedDict):
    a: int

class Extended(Base):
    b: str

def test_inherited_in(e: Extended, k: str):
    # Should narrow to all keys including inherited ones
    if k in e:
        reveal_type(k)  # E: revealed type: Literal['a', 'b']

def test_inherited_not_in(e: Extended, k: Literal['a', 'b', 'c']):
    if k not in e:
        reveal_type(k)  # E: revealed type: Literal['c']
    else:
        reveal_type(k)  # E: revealed type: Literal['a', 'b']
"#,
);

testcase!(
    test_typed_dict_contains_narrowing_empty,
    r#"
from typing import TypedDict, Literal, reveal_type

class Empty(TypedDict):
    pass

def test_empty_in(e: Empty, k: str):
    # Empty TypedDict - `in` check is always false, so type narrows to Never
    if k in e:
        reveal_type(k)  # E: revealed type: Never
    else:
        reveal_type(k)  # E: revealed type: str

def test_empty_not_in(e: Empty, k: str):
    # Empty TypedDict - `not in` check is always true, type is unchanged
    if k not in e:
        reveal_type(k)  # E: revealed type: str
    else:
        reveal_type(k)  # E: revealed type: Never
"#,
);

testcase!(
    test_illegal_unpacking_in_def,
    r#"
from typing import TypedDict
def f() -> dict: ...
X = TypedDict("X", {"k1": int, **f()})  # E: Unpacking is not supported
    "#,
);

testcase!(
    test_anonymous_typed_dict_missing_required_key,
    r#"
from typing import TypedDict

class TD(TypedDict):
    a: int
    b: bool

def f(td: TD) -> None: ...

td = {"a": 1000}
f(td=td)  # E: `b` is present in `TD` and absent in `<anonymous>`
    "#,
);

testcase!(
    test_anonymous_typed_dict_missing_not_required_key_ok,
    r#"
from typing import TypedDict

class TD(TypedDict, total=False):
    a: int
    b: bool

def f(td: TD) -> None: ...

td = {"a": 1000}
f(td=td)
    "#,
);

testcase!(
    test_anonymous_typed_dict_any_requiredness_ok,
    r#"
from typing import NotRequired, Required, TypedDict

class TD1(TypedDict):
    a: Required[int]

class TD2(TypedDict):
    a: NotRequired[int]

def f(td1: TD1, td2: TD2) -> None: ...

td = {"a": 1000}
f(td1=td, td2=td)
    "#,
);

testcase!(
    test_anonymous_typed_dict_any_extra_items_ok,
    r#"
from typing import ReadOnly, TypedDict

class TD1(TypedDict, closed=True):
    a: int

class TD2(TypedDict, extra_items=str):
    a: int

class TD3(TypedDict, extra_items=ReadOnly[str]):
    a: int

def f(td1: TD1, td2: TD2, td3: TD3) -> None: ...

td = {"a": 1000}
f(td1=td, td2=td, td3=td)
    "#,
);

testcase!(
    test_anonymous_typed_dict_value_subtype,
    r#"
from typing import TypedDict

class TD(TypedDict):
    a: int

def f(td: TD) -> None: ...

td = {"a": True}
f(td=td)
    "#,
);

testcase!(
    test_anonymous_typed_dict_check_extra_items,
    r#"
from typing import TypedDict

class TD1(TypedDict):
    a: int

class TD2(TypedDict, extra_items=str):
    a: int

def f(td1: TD1, td2: TD2) -> None: ...

td = {"a": 0, "b": "hi"}
f(
    td1=td,  # E: Field `b` is present in `<anonymous>` and absent in `TD1`
    td2=td,
)
    "#,
);
