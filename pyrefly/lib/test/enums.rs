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
    let fields = cls
        .fields()
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
assert_type(MyEnum.__PRIVATE, int)
assert_type(MyEnum.X.name, Literal["X"])
assert_type(MyEnum.X._name_, Literal["X"])
assert_type(MyEnum.X.value, int)
assert_type(MyEnum.X._value_, int)

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

# String literal doesn't match variable name
Color = Enum("C", 'RED', 'GREEN', 'BLUE')  # E: Expected string literal "Color"
"#,
);

testcase!(
    bug = "Enums with multiple inheritance are not iterable. Maybe a MRO / metaclass resolution issue?",
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
for e in E2: # E: Type `type[E2]` is not iterable
    assert_type(e, E2)
for e in E3: # E: Type `type[E3]` is not iterable
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
    Y = "FOO"  # E: Enum member `Y` has type `str`, must match the `_value_` attribute annotation of `int`
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
reveal_type(MyEnum.B)  # E: revealed type: nonmember[int]
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

// In 3.10 and lower versions, _magic_enum_attr is a different type than in 3.11+
testcase!(
    test_magic_enum_attr_3_10,
    TestEnv::new_with_version(PythonVersion::new(3, 10, 0)),
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
    test_magic_enum_attr_3_11,
    TestEnv::new_with_version(PythonVersion::new(3, 11, 0)),
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
    GREEN = "wrong" # E: Enum member `GREEN` has type `str`, must match the `_value_` attribute annotation of `int`
"#
    );
    env.add_with_path("pyi", "pyi.pyi", r#"
from enum import IntEnum

class Color(IntEnum):
    RED = ...
    GREEN = "wrong" # E: Enum member `GREEN` has type `str`, must match the `_value_` attribute annotation of `int`
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
    bug = "The RED = ... in pyi should be fine",
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
