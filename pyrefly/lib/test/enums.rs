/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::test::util::get_class;
use crate::test::util::mk_state;
use crate::testcase;

#[test]
fn test_fields() {
    let (handle, state) = mk_state(
        r#"
import enum
class E(enum.Enum):
    X = 1
    Y = 2
        "#,
    );
    let cls = get_class("E", &handle, &state);
    let bindings = state.transaction().get_bindings(&handle).unwrap();
    let class_fields = bindings.get_class_fields(cls.index()).unwrap();
    let fields = class_fields
        .names()
        .map(|f| f.as_str())
        .sorted()
        .collect::<Vec<_>>();
    assert_eq!(fields, vec!["X", "Y"]);
}

testcase!(
    test_enum_basic,
    r#"
from typing import assert_type, Literal
from enum import Enum

class MyEnum(Enum):
    X = 1
    Y = 2
    __PRIVATE = 3

assert_type(MyEnum.X, Literal[MyEnum.X])
assert_type(MyEnum["X"], Literal[MyEnum.X])
assert_type(MyEnum.__PRIVATE, int)  # E: Private attribute `__PRIVATE` cannot be accessed outside of its defining class
assert_type(MyEnum.X.name, Literal["X"])
assert_type(MyEnum.X._name_, Literal["X"])
assert_type(MyEnum.X.value, Literal[1])
assert_type(MyEnum.X._value_, Literal[1])

MyEnum["FOO"]  # E: Enum `MyEnum` does not have a member named `FOO`

def foo(member: str) -> None:
    assert_type(MyEnum[member], MyEnum)

def bar(member: int) -> None:
    MyEnum[member] # E: Enum `MyEnum` can only be indexed by strings

def foo(member: MyEnum) -> None:
    assert_type(member.name, str)
    assert_type(member.value, int)
    assert_type(member._value_, int)
"#,
);

testcase!(
    test_enum_class_value,
    r#"
from enum import Enum
from typing import assert_type, Literal, overload

class E(Enum):
    X = int

@overload
def f(x: Literal[E.X]) -> int: ...
@overload
def f(x: E) -> int | str | None: ...
def f(x) -> int | str | None: ...

assert_type(f(E.X), int)
"#,
);

testcase!(
    test_enum_meta,
    r#"
from typing import assert_type, Literal
from enum import EnumMeta

class CustomEnumType(EnumMeta):
    pass

class CustomEnum(metaclass=CustomEnumType):
    pass

class Color(CustomEnum):
    RED = 1
    GREEN = 2
    BLUE = 3

assert_type(Color.RED, Literal[Color.RED])
"#,
);

testcase!(
    test_enum_functional,
    r#"
from typing import assert_type, Literal
from enum import Enum

Color2 = Enum('Color2', 'RED', 'GREEN', 'BLUE')
Color3 = Enum('Color3', ['RED', 'GREEN', 'BLUE'])
Color4 = Enum('Color4', ('RED', 'GREEN', 'BLUE'))
Color5 = Enum('Color5', 'RED, GREEN, BLUE')
Color6 = Enum('Color6', 'RED GREEN BLUE')
Color7 = Enum('Color7', [('RED', 1), ('GREEN', 2), ('BLUE', 3)])
Color8 = Enum('Color8', (('RED', 1), ('GREEN', 2), ('BLUE', 3)))
Color9 = Enum('Color9', {'RED': 1, 'GREEN': 2, 'BLUE': 3})

assert_type(Color2.RED, Literal[Color2.RED])
assert_type(Color3.RED, Literal[Color3.RED])
assert_type(Color4.RED, Literal[Color4.RED])
assert_type(Color5.RED, Literal[Color5.RED])
assert_type(Color6.RED, Literal[Color6.RED])
assert_type(Color7.RED, Literal[Color7.RED])
assert_type(Color8.RED, Literal[Color8.RED])
assert_type(Color9.RED, Literal[Color9.RED])
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2874
testcase!(
    test_enum_functional_name_mismatch,
    r#"
from typing import Literal, assert_type
from enum import Enum

_dvistate = Enum("DviState", "pre outer inpage post_post finale")  # E: Expected string literal "_dvistate"

assert_type(_dvistate.pre, Literal[_dvistate.pre])
assert_type(_dvistate.post_post, Literal[_dvistate.post_post])
"#,
);

testcase!(
    test_iterate,
    r#"
from typing import assert_type
from enum import Enum, StrEnum

class E1(Enum):
    X = 1

class E2(str, Enum):
    X = "1"

class E3(StrEnum):
    X = "1"

for e in E1:
    assert_type(e, E1)
for e in E2:
    assert_type(e, E2)
for e in E3:
    assert_type(e, E3)

    "#,
);

testcase!(
    test_value_annotation,
    r#"
from enum import Enum, member, auto

class MyEnum(Enum):
    _value_: int
    V = member(1)
    W = auto()
    X = 1
    Y = "FOO"  # E: Enum member `Y` has type `Literal['FOO']`, must match the `_value_` attribute annotation of `int`
    Z = member("FOO")  # E: Enum member `Z` has type `str`, must match the `_value_` attribute annotation of `int`

    def get_value(self) -> int:
        if self.value > 0:
            return self.value
        else:
            return self._value_
"#,
);

testcase!(
    test_infer_value,
    r#"
from enum import Enum
from typing import assert_type

class MyEnum(Enum):
    X = 1
    Y = "foo"
def test(e: MyEnum):
    # the inferred type use promoted types, for performance reasons
    assert_type(e.value, int | str)
"#,
);

testcase!(
    test_mutate_value,
    r#"
from enum import Enum
class MyEnumAnnotated(Enum):
    _value_: int
    X = 1
class MyEnumUnannotated(Enum):
    X = 1
def mutate(ea: MyEnumAnnotated, eu: MyEnumUnannotated) -> None:
    ea._value_ = 2  # Allowed for now, because it must be permitted in `__init__`
    ea.value = 2  # E: Cannot set field `value`
    eu._value_ = 2  # Allowed for now, because it must be permitted in `__init__`
    eu.value = 2  # E: Cannot set field `value`
"#,
);

testcase!(
    test_value_annotation_irrelevant_for_getattr,
    r#"
from enum import Enum

class MyEnum(Enum):
    X = 1
    Y = "FOO"

    # We won't be resolving the type of `_value_` through `__getattr__`
    def __getattr__(self, name: str) -> int: ...
"#,
);

testcase!(
    test_enum_member,
    r#"
from enum import Enum, nonmember, member
from typing import Literal, reveal_type, assert_type

class MyEnum(Enum):
    A = 1
    B = nonmember(2)
    @member
    def C(self) -> None: pass
    def D(self) -> None: pass

reveal_type(MyEnum.A)  # E: revealed type: Literal[MyEnum.A]
reveal_type(MyEnum.B)  # E: revealed type: int
reveal_type(MyEnum.C)  # E: revealed type: Literal[MyEnum.C]
reveal_type(MyEnum.D)  # E: revealed type: (self: MyEnum) -> None
"#,
);

testcase!(
    test_member_with_explicit_annotation,
    r#"
from typing import assert_type, Literal
from enum import Enum

class MyEnum(Enum):
    X: float = 5  # E: Enum member `X` may not be annotated directly. Instead, annotate the `_value_` attribute

assert_type(MyEnum.X, Literal[MyEnum.X])
assert_type(MyEnum.X.value, float)
"#,
);

testcase!(
    test_value_of_union_of_enum_literals,
    r#"
from typing import Literal
from enum import Enum
class E(Enum):
    X = 1
    Y = 2
def f(e: Literal[E.X, E.Y]) -> int:
    return e.value
    "#,
);

testcase!(
    test_enum_union_simplification,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E1(Enum):
    X = 1
    Y = 2
class E2(Enum):
    X = 1
    Y = 2
    Z = 3
def f(test: bool):
    # union of all possible enum members simplifies to the enum class
    e1 = E1.X if test else E1.Y
    assert_type(e1, E1)

    # this doesn't simplify because not all members are included
    e2 = E2.X if test else E2.Y
    assert_type(e2, Literal[E2.X, E2.Y])
    "#,
);

testcase!(
    test_enum_subset_of_union,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E1(Enum):
    X = 1
    Y = 2
class E2(Enum):
    X = 1
    Y = 2
    Z = 3
def f(test: bool, e1: E1, e2: E2):
    x: Literal[E1.X, E1.Y] = e1
    y: Literal[E1.X, E1.Y, 1] = e1
    z: Literal[E2.X, E2.Y] = e2  # E: `E2` is not assignable to `Literal[E2.X, E2.Y]`
    "#,
);

testcase!(
    test_flag,
    r#"
from enum import Flag
from typing import assert_type

class MyFlag(Flag):
    X = 1
    Y = 2

def foo(f: MyFlag) -> None:
    if f == MyFlag.X:
        pass
    else:
        assert_type(f, MyFlag)
"#,
);

testcase!(
    test_enum_instance_only_attr,
    r#"
from typing import assert_type, Any
from enum import Enum

class MyEnum(Enum):
    X = "foo"
    Y: int
    Z = "bar"

assert_type(MyEnum.Y, int)

for x in MyEnum:
    assert_type(x.value, str)  # Y is not an enum member
"#,
);

testcase!(
    test_generic_enum,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E[T](Enum):  # E: Enums may not be generic
    X = 1
# Even though a generic enum is an error, we still want to handle it gracefully.
assert_type(E.X, Literal[E.X])
    "#,
);

testcase!(
    test_enum_dunder_members,
    r#"
from enum import Enum, EnumMeta
class MyEnum(Enum):
    X = 1
    Y = "FOO"
MyEnum.__members__
"#,
);

testcase!(
    test_enum_extend_final,
    r#"
from enum import Enum
class A(Enum): pass

class B(Enum):
    X = 1

class C(A):
    X = 1

class D(B): # E: Cannot extend final class `B`
    pass
"#,
);

testcase!(
    test_enum_name,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E(Enum):
    X = 1
    def get_name(self) -> str:
        if self.name:
            return self.name
        else:
            return self._name_
# Even though a generic enum is an error, we still want to handle it gracefully.
assert_type(E.X._name_, Literal["X"])
assert_type(E.X.name, Literal["X"])
    "#,
);

testcase!(
    test_enum_union,
    r#"
from typing import assert_type, Literal
from enum import Enum

class MyEnum(Enum):
    X = 1
    Y = 2

def f(cond: bool, a: MyEnum, b: Literal[MyEnum.X]):
    if cond:
        return a
    else:
        return b

assert_type(f(True, MyEnum.X, MyEnum.X), MyEnum)
"#,
);

testcase!(
    test_enum_override_value,
    r#"
from enum import Enum
from typing import assert_type

class MyIntEnum(int, Enum):
    TWENTYSIX = '1a', 16
    value: int

assert_type(MyIntEnum.TWENTYSIX.value, int)
"#,
);

// In 3.10 and lower versions, _magic_enum_attr is a different type than in 3.11+
testcase!(
    test_magic_enum_attr_3_10,
    TestEnv::new_with_version(PythonVersion::new(3, 10, 0)),
    r#"
from typing_extensions import assert_type, Any
import enum
class E(enum.Enum):
    _value_: int
    E0 = 0
    E1 = 1
    @enum._magic_enum_attr
    def foo(self) -> str: ...
e = E.E0
assert_type(e.foo, Any)
    "#,
);

testcase!(
    test_magic_enum_attr_3_11,
    TestEnv::new_with_version(PythonVersion::new(3, 11, 0)),
    r#"
from typing_extensions import assert_type
import enum
class E(enum.Enum):
    _value_: int
    E0 = 0
    E1 = 1
    @enum._magic_enum_attr
    def foo(self) -> str: ...
e = E.E0
assert_type(e.foo, str)
    "#,
);

testcase!(
    test_enum_literal,
    r#"
import enum
from typing import assert_type, Literal

class A(enum.IntEnum):
    B = 'positional or keyword'

    # right now, we don't check the type of the enum member if the enum class defines `__new__`
    def __new__(cls, description):
        value = len(cls.__members__)
        member = int.__new__(cls, value)
        return member

assert_type(A.B, Literal[A.B])
    "#,
);

testcase!(
    test_intenum_numeric_tower,
    r#"
import enum
from typing import assert_type

class Period(enum.IntEnum):
    DAY = 24

def takes_float(x: float) -> float:
    return x

assert_type(takes_float(Period.DAY), float)
assert_type(takes_float(24), float)
assert_type(takes_float(24.0), float)
    "#,
);

// This used to trigger a false positive where we thought the metaclass inheriting
// Any meant it was an enum metaclass, see https://github.com/facebook/pyrefly/issues/622
testcase!(
    test_metaclass_subtype_of_any_is_not_enum_metaclass,
    r#"
from typing import Any
class CustomMetaclass(Any):
    pass
class C[T](metaclass=CustomMetaclass):  # Ok - was a false positive
    x: T
    "#,
);

fn env_enum_dots() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path("py", "py.py", r#"
from enum import IntEnum

class Color(IntEnum):
    RED = ... # E: Enum member `RED` has type `Ellipsis`, must match the `_value_` attribute annotation of `int`
    GREEN = "wrong" # E: Enum member `GREEN` has type `Literal['wrong']`, must match the `_value_` attribute annotation of `int`
"#
    );
    env.add_with_path("pyi", "pyi.pyi", r#"
from enum import IntEnum

class Color(IntEnum):
    RED = ...
    GREEN = "wrong" # E: Enum member `GREEN` has type `Literal['wrong']`, must match the `_value_` attribute annotation of `int`
"#
    );
    env
}

testcase!(
    test_enum_descriptor,
    r#"
from enum import IntEnum
from typing import Callable, assert_type

class classproperty[_TClass, _TReturnType]:
    fget: Callable[[_TClass], _TReturnType]
    def __init__(self, f: Callable[[_TClass], _TReturnType]) -> None: ...
    def __get__(self, obj: _TClass | None, cls: _TClass) -> _TReturnType: ...

class Foo(IntEnum):
    X = 1
    @classproperty
    def Y(cls) -> list[Foo]:
        return [Foo.X]

# descriptors are not enum members
assert_type(Foo.Y, list[Foo])
"#,
);

testcase!(
    test_enum_value_dots_pyi,
    env_enum_dots(),
    r#"
import py
import pyi

from typing import assert_type, Literal
assert_type(py.Color.RED, Literal[py.Color.RED])
assert_type(pyi.Color.RED, Literal[pyi.Color.RED])
"#,
);

testcase!(
    test_empty_functional_def,
    r#"
from enum import Enum
E = Enum('E', [])
    "#,
);

testcase!(
    test_empty_enum,
    r#"
from typing import Any, assert_type
from enum import Enum
class EmptyEnum(Enum):
    # in real code there might be dynamic logic here, e.g. `vars()[key] = value`.
    pass
def test(x: EmptyEnum):
    assert_type(x.value, Any)
    "#,
);

testcase!(
    test_enum_iter,
    r#"
from enum import Enum
from typing import TypeVar

class MyEnum(Enum):
    A = "a"
    B = "b"

T_Enum = TypeVar("T_Enum", bound=Enum)

def get_labels(enum_cls: type[T_Enum]) -> list[str]:
    return [e.name for e in enum_cls]
    "#,
);

testcase!(
    test_enum_type_getitem,
    r#"
from enum import Enum
from typing import TypeVar, assert_type

class Color(Enum):
    RED = "red"
    BLUE = "blue"

def accepts_base(cls: type[Enum], key: str) -> None:
    assert_type(cls[key], Enum)

def accepts_specific(cls: type[Color], key: str) -> None:
    assert_type(cls[key], Color)

T_Enum = TypeVar("T_Enum", bound=Enum)

def accepts_generic(cls: type[T_Enum], key: str) -> None:
    assert_type(cls[key], T_Enum)

def bad_key(cls: type[Enum]) -> None:
    cls[0]  # E: Enum type `type[Enum]` can only be indexed by strings
"#,
);

testcase!(
    test_mixin_datatype,
    r#"
from enum import Enum
from typing import assert_type, Literal

class A(float, Enum):
    X = 1

class FloatEnum(float, Enum):
    pass
class B(FloatEnum):
    X = 1

assert_type(A.X.value, float)
assert_type(B.X.value, float)
    "#,
);

testcase!(
    test_override_value_prop,
    r#"
from enum import Enum
from typing import assert_type, Literal
class E(Enum):
    X = 1
    @property
    def value(self) -> str: ...
assert_type(E.X._value_, Literal[1])
assert_type(E.X.value, str)
    "#,
);

testcase!(
    test_auto,
    r#"
from enum import auto, Enum, StrEnum
from typing import assert_type
class E1(Enum):
    X = auto()
class E2(StrEnum):
    X = auto()
class E3(str, Enum):
    X = auto()
class E4(Enum):
    X = (auto(),)
assert_type(E1.X.value, int)
assert_type(E2.X.value, str)
assert_type(E3.X.value, str)
assert_type(E4.X.value, tuple[int])
    "#,
);

testcase!(
    test_auto_generate_next_value,
    r#"
from enum import auto, Enum, StrEnum
from typing import assert_type

# Custom _generate_next_value_ returning float
class FloatEnum(Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[float]) -> float: ...
    X = auto()
    Y = auto()

assert_type(FloatEnum.X.value, float)
assert_type(FloatEnum.Y.value, float)

# StrEnum auto() generates str values
class Color(StrEnum):
    RED = auto()
    GREEN = auto()

assert_type(Color.RED.value, str)

# Mixin type takes priority over _generate_next_value_
class BytesEnum(bytes, Enum):
    X = auto()

assert_type(BytesEnum.X.value, bytes)

# Generic instance access should also use auto inference
def check_float_enum(e: FloatEnum) -> None:
    assert_type(e.value, float)
    "#,
);

testcase!(
    test_callable_nonmember,
    r#"
from enum import Enum
from typing import Callable

class InclusionLevel(Enum):
    A = 1
    B = 2
    C = 3

    def is_included(self):
        return self.value >  self.B.value

x: Callable[[InclusionLevel], bool] = InclusionLevel.is_included
    "#,
);

testcase!(
    test_callable_enum,
    r#"
from enum import Enum
from typing import assert_type

class MyCallable:
    def __call__(self) -> int:
        return 42

class E1(MyCallable, Enum):
    pass

class E2(E1):
    X = 1

assert_type(E2.X(), int)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/755
testcase!(
    test_access_value_on_mixed_type_enum,
    r#"
from enum import Enum

class StrEnum(str, Enum):
    FOO = "FOO"
    DEFAULT = "DEFAULT"

    @classmethod
    def normalize(cls, val: str) -> str:
        try:
            return cls(val).value
        except ValueError:
            return cls.DEFAULT.value

class IntEnum(int, Enum):
    FOO = 1
    DEFAULT = 0

    @classmethod
    def normalize(cls, val: int) -> int:
        try:
            return cls(val).value
        except ValueError:
            return cls.DEFAULT.value
    "#,
);

testcase!(
    test_enum_call_uses_metaclass_signature,
    r#"
from enum import Enum
from typing import Callable, assert_type

class SeFileType(Enum):
    ALL = ("a", "all files")
    REGULAR = ("f", "regular file")
    DIRECTORY = ("d", "directory")

    def __new__(cls, code: str, description: str) -> "SeFileType":
        obj = object.__new__(cls)
        obj._value_ = code
        return obj

    @classmethod
    def from_code(cls, code: str) -> "SeFileType":
        assert_type(cls(code), SeFileType)
        return cls(code)

assert_type(SeFileType("a"), SeFileType)
constructor: Callable[[str], SeFileType] = SeFileType
    "#,
);

testcase!(
    test_enum_call_with_self_type,
    r#"
from enum import Enum
from typing import Self, assert_type

class SeFileType(Enum):
    ALL = ("a", "all files")
    REGULAR = ("f", "regular file")

    def __new__(cls, code: str, description: str) -> "SeFileType":
        obj = object.__new__(cls)
        obj._value_ = code
        return obj

    @classmethod
    def from_code(cls, code: str) -> Self:
        assert_type(cls, type[Self])
        assert_type(cls(code), Self)
        return cls(code)
    "#,
);

testcase!(
    test_enum_alias,
    r#"
from typing import assert_type, Literal
from enum import Enum

class TrafficLight(Enum):
    YELLOW = 3
    AMBER = YELLOW  # Alias for YELLOW

assert_type(TrafficLight.AMBER, Literal[TrafficLight.YELLOW])
    "#,
);

testcase!(
    test_illegal_unpacking_in_def,
    r#"
from enum import Enum
def f() -> dict: ...
X = Enum("X", {'FOO': 1, **f()})  # E: Unpacking is not supported
    "#,
);

testcase!(
    test_enum_classmethod,
    r#"
from enum import Enum

class Foo(str, Enum):
    A = "a"
    B = "b"

    @classmethod
    def from_dict(cls, default=None):
        if default is None:
            default = cls.A
        return cls(default)
Foo.from_dict({})
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2583
// StrEnum defines its own `_value_: str`, so the guard on the special-case `_value_` path
// must include StrEnum (and IntEnum) in addition to `enum.Enum`.
testcase!(
    test_strenum_value_literal_type,
    r#"
from enum import StrEnum, Enum
from typing import assert_type, Literal

class Foo(StrEnum):
    X = "x"

class Bar(str, Enum):
    Y = "y"

def take_literal(z: Literal["x", "y"]) -> None: ...

# StrEnum: specific literal gives literal type
assert_type(Foo.X.value, Literal["x"])
# str, Enum mixin: specific literal correctly gives literal type
assert_type(Bar.Y.value, Literal["y"])

# Generic instance access should give the mixed-in type
def test(foo: Foo, bar: Bar) -> None:
    assert_type(foo.value, str)
    assert_type(bar.value, str)
    take_literal(Foo.X.value)
    take_literal(Bar.Y.value)
    "#,
);

// When a member's value type doesn't match the mixin (e.g. int value in a str-mixin enum),
// the mixin's `__new__` coerces the value at runtime (e.g. `str(42)` → `"42"`),
// so `.value` returns the mixin type.
testcase!(
    test_mixin_value_type_mismatch,
    r#"
from enum import Enum
from typing import assert_type, Literal

class Bad(str, Enum):
    X = 42

assert_type(Bad.X.value, str)
    "#,
);

// When `__new__` converts the value type (e.g. int → str), the literal is only
// preserved if it's a subtype of the mixin. Otherwise we fall back to the mixin type.
testcase!(
    test_mixin_new_converts_type,
    r#"
from enum import Enum
from typing import assert_type, Literal

class E(str, Enum):
    # String literal is a subtype of str, so the literal is preserved.
    A = "hello"
    # Int literal is NOT a subtype of str; str.__new__ coerces it at runtime,
    # so `.value` falls back to the mixin type.
    B = 42

assert_type(E.A.value, Literal["hello"])
assert_type(E.B.value, str)
    "#,
);

// When `__new__` is defined, it can rewrite `_value_` at runtime, so the raw RHS type
// is unreliable. Without a mixin or explicit `_value_` annotation, we fall back to `Any`
// since we can't infer what `__new__` assigns.
testcase!(
    test_new_rewrites_value,
    r#"
from enum import Enum
from typing import assert_type, Literal, Any

class SeFileType(Enum):
    ALL = ("a", "all files")
    REGULAR = ("f", "regular file")

    def __new__(cls, code: str, description: str) -> "SeFileType":
        obj = object.__new__(cls)
        obj._value_ = code
        return obj

assert_type(SeFileType.ALL.value, Any)
    "#,
);

// When `__new__` is defined AND a mixin is present, fall back to the mixin type.
testcase!(
    test_new_with_mixin,
    r#"
from enum import Enum
from typing import assert_type

class Planet(float, Enum):
    MERCURY = (3.303e+23, 2.4397e6)

    def __new__(cls, mass: float, radius: float) -> "Planet":
        obj = float.__new__(cls, mass)
        obj._value_ = mass
        return obj

assert_type(Planet.MERCURY.value, float)
    "#,
);

// A non-data-type mixin (no __new__) should not affect .value type inference.
testcase!(
    test_mixin_not_data_type,
    r#"
from enum import Enum, IntEnum
from typing import assert_type, Literal

class Meta:
    def some_method(self) -> str:
        return "hello"

class MyEnum(Meta, Enum):
    pass

class MyIntEnum(Meta, IntEnum):
    pass

class Foo(MyEnum):
    bar = 1

class Bar(MyIntEnum):
    foo = 1

assert_type(Foo.bar.value, Literal[1])
assert_type(Bar.foo.value, Literal[1])
    "#,
);

// A data type mixin (str) should still work when combined with a non-data-type mixin.
testcase!(
    test_mixin_data_type_with_regular_mixin,
    r#"
from enum import Enum
from typing import assert_type, Literal

class Meta:
    pass

class MyStrEnum(Meta, str, Enum):
    pass

class Baz(MyStrEnum):
    x = "hello"

assert_type(Baz.x.value, Literal["hello"])
    "#,
);

// A subclass of a data type (e.g. MyStr(str)) inherits __new__ and should
// still be treated as a data type mixin.
testcase!(
    test_mixin_inherited_data_type,
    r#"
from enum import auto, Enum
from typing import assert_type

class MyStr(str):
    pass

class MyInt(int):
    pass

class StrEnum(MyStr, Enum):
    X = auto()

class IntEnum2(MyInt, Enum):
    Y = auto()

assert_type(StrEnum.X.value, MyStr)
assert_type(IntEnum2.Y.value, MyInt)
    "#,
);

fn frozen_enum_members_env() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from enum import Enum
from typing import Self
class E(Enum):
    A = frozenset({1})
    B = frozenset({2})
    @classmethod
    def from_ord(cls) -> list[Self]:
        return [v for v in cls]
    "#,
    )
}

testcase!(
    test_frozen_enum_members_cross_module_iteration,
    frozen_enum_members_env(),
    r#"
from foo import E
def f() -> None:
    xs = E.from_ord()
    _ = [x for x in xs if x != E.A]
    "#,
);
