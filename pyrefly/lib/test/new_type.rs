/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_new_type_simple,
    r#"
from typing import NewType, assert_type

UserId = NewType("UserId", int)
UserId("user")  # E: Argument `Literal['user']` is not assignable to parameter `_x` with type `int` in function `UserId.__new__`
u1: UserId = 42 # E: `Literal[42]` is not assignable to `UserId`
u2: UserId = UserId(42)

assert_type(UserId(5) + 1, int)

isinstance(u2, UserId) # E: NewType `UserId` not allowed in isinstance

class UserIdDerived(UserId): # E: Subclassing a NewType not allowed
    pass
     "#,
);

testcase!(
    test_new_type_naming,
    r#"
from typing import NewType 

GoodName = NewType("BadName", int) # E: Expected string literal "GoodName"  

GoodNewType1 = NewType("GoodNewType1", list)  

GoodNewType2 = NewType("GoodNewType2", GoodNewType1) 

nt1: GoodNewType1[int] # E: Expected 0 type arguments for `GoodNewType1`, got 1

     "#,
);

testcase!(
    test_new_type_generic,
    r#"
from typing import NewType, TypeVar, Hashable, Literal

BadNewType1 = NewType("BadNewType1", int | str) # E: Second argument to NewType is invalid

T = TypeVar("T")
BadNewType2 = NewType("BadNewType2", list[T])  # E: Second argument to NewType cannot be an unbound generic

BadNewType3 = NewType("BadNewType3", Hashable) # E: Second argument to NewType cannot be a protocol

BadNewType4 = NewType("BadNewType4", Literal[7]) # E: Second argument to NewType is invalid
     "#,
);

testcase!(
    test_new_type_wrong_arity,
    r#"
from typing import NewType 
UserId = NewType("UserId", int, int) # E: Expected 2 positional arguments, got 3
UserId = NewType("UserId") # E: Missing argument `tp`
userId = NewType() # E: Missing argument `name` # E: Missing argument `tp`
     "#,
);

testcase!(
    test_new_type_not_allowed,
    r#"
from typing import NewType, Generic, Protocol, TypedDict, TypeVar, Any

class TD1(TypedDict):
    a: int

T = TypeVar("T")

BadNewType1 = NewType("BadNewType1", TD1)  # E: Second argument to NewType is invalid

BadNewType2 = NewType("BadNewType2", TypedDict)  # E: Second argument to NewType is invalid

BadNewType3 = NewType("BadNewType3", Protocol)  # E: Second argument to NewType is invalid

BadNewType4 = NewType("BadNewType4", Generic[T])  # E: Second argument to NewType is invalid

BadNewType5 = NewType("BadNewType5", Any)  # E: Second argument to NewType is invalid
     "#,
);

testcase!(
    test_new_type_as_alias,
    r#"
from typing import NewType, TypeAlias

class R:
    ValueType = NewType("ValueType", int)
    V: TypeAlias = ValueType

    def test(self, v: V) -> int:
        return v
     "#,
);

testcase!(
    test_new_type_tuple,
    r#"
from typing import NewType
Foo = NewType("Foo", tuple[int, int])

Foo((1, 2))  # OK
Foo((1, 2, 3))  # E: Argument `tuple[Literal[1], Literal[2], Literal[3]]` is not assignable to parameter `_x` with type `tuple[int, int]` in function `Foo.__new__`
     "#,
);

testcase!(
    test_new_type_inherits_tuple,
    r#"
from typing import NewType
class A(tuple[int, int]):
    pass
X = NewType("X", A)
X((0, 0))  # E: Argument `tuple[Literal[0], Literal[0]]` is not assignable to parameter `_x` with type `A` in function `X.__new__`
    "#,
);

testcase!(
    test_new_type_is_not_type,
    r#"
from typing import Any, NewType
Foo = NewType("Foo", int)
x: type = Foo  # E: `type[Foo]` is not assignable to `type`
y: type[Any] = Foo  # E: `type[Foo]` is not assignable to `type[Any]`
    "#,
);

testcase!(
    test_new_type_type_argument,
    r#"
from typing import NewType, Type

Thing = NewType("Thing", int)
ThingType = type[Thing]  # E: NewType `Thing` is not a class and cannot be used with `type` or `Type`
OtherThingType = Type[Thing]  # E: NewType `Thing` is not a class and cannot be used with `type` or `Type`

mapping: dict[int, ThingType] = {1: Thing}  # E: `dict[int, type[Thing]]` is not assignable to `dict[int, type[Unknown]]`

def func(x: ThingType) -> None: ...
func(Thing)  # E: Argument `type[Thing]` is not assignable to parameter `x` with type `type[Unknown]` in function `func`
    "#,
);

testcase!(
    test_new_type_runtime_attrs,
    r#"
from typing import NewType
Foo = NewType("Foo", int)
Foo.__getattribute__
Foo.__repr__
Foo.mro()  # E: Object of class `object` has no attribute `mro`
    "#,
);

testcase!(
    test_tuple,
    r#"
from typing import NewType
Inty = NewType("Inty", tuple[int, int])
x = Inty((1,2))
def f(v: tuple[int, int]):
    pass
f(x)
    "#,
);
