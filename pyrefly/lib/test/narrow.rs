/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_is,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    assert_type(x, str | None)
    "#,
);

testcase!(
    test_truthy_falsy,
    r#"
from typing import assert_type, Literal
def f(x: str | None, y: bool):
    if x:
        assert_type(x, str)
    if y:
        assert_type(y, Literal[True])
    else:
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_eq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x == None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_neq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x != None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is not None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_if_else,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    else:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_subtype,
    r#"
from typing import assert_type
class A: pass
class B(A): pass
def f(x: type[A]):
    if x is B:
        assert_type(x, type[B])
    "#,
);

testcase!(
    test_is_never,
    r#"
from typing import assert_type, Never
def f(x: str):
    if x is None:
        assert_type(x, Never)
    "#,
);

testcase!(
    test_is_not_bool_literal,
    r#"
from typing import assert_type, Literal, Never
def f1(x: bool):
    if x is not True:
        assert_type(x, Literal[False])
def f2(x: Literal[True] | str):
    if x is not True:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not_enum_literal,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
def f1(x: Literal[E.X, E.Y]):
    if x is not E.X:
        assert_type(x, Literal[E.Y])
def f2(x: E | int):
    if x is not E.X:
        assert_type(x, Literal[E.Y] | int)
    "#,
);

testcase!(
    test_tri_enum,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
    Z = 3
def f(x: E):
    if x is E.X:
       assert_type(x, Literal[E.X])
    elif x is E.Y:
       assert_type(x, Literal[E.Y])
    else:
       assert_type(x, Literal[E.Z])
    "#,
);

testcase!(
    test_is_classdef,
    r#"
from typing import assert_type
class A: pass
class B: pass
def f1(x: type[A] | type[B]):
    if x is A:
        assert_type(x, type[A])
    else:
        # Note that we cannot narrow to `type[B]` here, as `type` is covariant and `x` may be a
        # subtype of `A`.
        assert_type(x, type[A] | type[B])
    "#,
);

testcase!(
    test_and,
    r#"
from typing import assert_type, Literal, Never
def f(x: bool | None):
    if x is True and x is None:
        assert_type(x, Never)
    else:
        assert_type(x, bool | None)
    "#,
);

testcase!(
    test_and_multiple_vars,
    r#"
from typing import assert_type, Literal
def f(x: bool | None, y: bool | None):
    if x is True and y is False:
        assert_type(x, Literal[True])
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_or,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if x == True or x is None:
        assert_type(x, Literal[True] | None)
    else:
        assert_type(x, Literal[False])
    "#,
);

testcase!(
    test_elif,
    r#"
from typing import assert_type
def f(x: str | None, y: int | None):
    if x is None:
        assert_type(x, None)
        assert_type(y, int | None)
    elif y is None:
        assert_type(x, str)
        assert_type(y, None)
    else:
        assert_type(x, str)
        assert_type(y, int)
    "#,
);

testcase!(
    test_not,
    r#"
from typing import assert_type
def f(x: str | None):
    if not x is None:
        assert_type(x, str)
    else:
        assert_type(x, None)
    if not x:
        assert_type(x, str | None)
    else:
        assert_type(x, str)
    "#,
);

testcase!(
    test_exit,
    r#"
from typing import assert_type
import sys
import os
def test_sys_exit(x: str | None):
    if not x:
        sys.exit(1)
    assert_type(x, str)
def test_exit(x: str | None):
    if not x:
        exit(1)
    assert_type(x, str)
def test_quit(x: str | None):
    if not x:
        quit(1)
    assert_type(x, str)
def test_os_exit(x: str | None):
    if not x:
        os._exit(1)
    assert_type(x, str)
    "#,
);

testcase!(
    test_not_and,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if not (x is True and x is None):
        assert_type(x, Literal[False] | bool | None)
    "#,
);

testcase!(
    test_assert,
    r#"
from typing import assert_type
def f(x: str | None):
    assert x is not None
    assert_type(x, str)
    "#,
);

testcase!(
    test_while_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    assert_type(x, None)
    x = f()
    assert_type(x, str | None)
else:
    assert_type(x, str)
assert_type(x, str)
    "#,
);

testcase!(
    test_while_break,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    break
assert_type(x, str | None)
    "#,
);

testcase!(
    test_while_break_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    if f():
        break
else:
    assert_type(x, str)
assert_type(x, str | None)
    "#,
);

testcase!(
    bug = "Unwanted EXPECTED error",
    test_while_overwrite,
    r#"
from typing import assert_type, Literal
def f() -> str | None: ...
x = f()
while x is None:  # E: EXPECTED Literal[42] | str <: str | None
    if f():
        x = 42
        break
assert_type(x, Literal[42] | str)
    "#,
);

testcase!(
    test_while_narrow,
    r#"
from typing import assert_type, Literal, reveal_type
def test(x: bool, z: bool):
    while x:
        assert_type(x, Literal[True])
    while y := z:
        assert_type(y, Literal[True])
        assert_type(z, Literal[True])
    "#,
);
testcase!(
    test_nested_function,
    r#"
from typing import assert_type
def foo(x: int | None) -> None:
    def include():
        if x is not None:
            assert_type(x, int)
    "#,
);

testcase!(
    test_multiple_is,
    r#"
from typing import assert_type, Never
def f(x: bool | None, y: bool | None):
    if x is None is None:
        assert_type(x, None)
    if y is None is True:
        assert_type(y, Never)
    "#,
);

testcase!(
    test_class_body,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
class C:
    if x is None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_walrus_target,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if x := f():
    assert_type(x, str)
    "#,
);

testcase!(
    test_walrus_value,
    r#"
from typing import assert_type
def f(x: int | None):
    if y := x:
        assert_type(x, int)
        assert_type(y, int)
    "#,
);

testcase!(
    test_walrus_comparison,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if (x := f()) is None:
    assert_type(x, None)
    "#,
);

testcase!(
    test_walrus_comprehension_if_simple,
    r#"
from typing import assert_type
def f(xs: list[int | None]):
    ys = [y111 for x in xs if (y111 := x)]
    assert_type(ys, list[int])
    "#,
);

testcase!(
    test_walrus_comprehension_if_function,
    r#"
from typing import assert_type
def get_y(x: int | None) -> int | None:
    return x
def f(xs: list[int | None]):
    ys = [y111 for x in xs if (y111 := get_y(x))]
    assert_type(ys, list[int])
    "#,
);

testcase!(
    test_walrus_generator_if,
    r#"
from typing import Sequence
def foo(x: int) -> int | None:
    return (x + 5) if x % 2 else None
foos: Sequence[int] = tuple(
    maybe_foo
    for x in range(10)
    if (maybe_foo := foo(x)) is not None
)
    "#,
);

testcase!(
    test_walrus_ternary,
    r#"
from typing import assert_type
def get_y(x: int | None) -> int | None:
    return x
def f(x: int | None):
    val = y if (y := get_y(x)) else 0
    assert_type(val, int)
    "#,
);

testcase!(
    test_match_enum_fallback,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E(Enum):
    X = 1
    Y = 2
    Z = 3
def f(e: E):
    match e:
        case E.X:
            assert_type(e, Literal[E.X])
        case E.Y:
            assert_type(e, Literal[E.Y])
        case _:
            assert_type(e, Literal[E.Z])
    "#,
);

testcase!(
    test_match_or,
    r#"
from typing import assert_type, Literal
def f(e: bool | None):
    match e:
        case True | None:
            assert_type(e, Literal[True] | None)
        case _:
            assert_type(e, Literal[False])
    "#,
);

testcase!(
    test_ternary,
    r#"
from typing import assert_type
def f(x: str | None, y: int):
    z = x if x else y
    assert_type(x, str | None)
    assert_type(y, int)
    assert_type(z, str | int)
    "#,
);

testcase!(
    test_is_supertype,
    r#"
from typing import Literal, assert_type
import enum
class E(enum.Enum):
    X = 1
def f(x: Literal[E.X], y: E):
    if x is y:
        assert_type(x, Literal[E.X])
    "#,
);

testcase!(
    test_isinstance,
    r#"
from typing import assert_type
def f(x: str | int):
    if isinstance(x, str):
        assert_type(x, str)
    else:
        assert_type(x, int)
    "#,
);

testcase!(
    test_isinstance_union,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, str | int):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    test_isinstance_of_tuple,
    r#"
from typing import assert_type
def f(x):
    if isinstance(x, tuple | int):
        assert_type(x, tuple | int)
        if isinstance(x, tuple):
            assert_type(x, tuple)
        else:
            assert_type(x, int)
"#,
);

testcase!(
    test_isinstance_of_none,
    r#"
from typing import assert_type
def f(x):
    if isinstance(x, None | int):
        assert_type(x, None | int)
        if isinstance(x, int):
            assert_type(x, int)
        else:
            assert_type(x, None)

def g(x):
    isinstance(x, None) # E: `None` is not assignable to parameter
"#,
);

testcase!(
    test_isinstance_tuple,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, (str, int)):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    test_isinstance_type,
    r#"
from typing import assert_type
def f(x: object, y: type[str]) -> None:
    if isinstance(x, y):
        assert_type(x, str)
    "#,
);

testcase!(
    test_issubclass_union,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
def f(x: type[A | B | C]):
    if issubclass(x, A | B):
        assert_type(x, type[A] | type[B])
    else:
        assert_type(x, type[C])
    "#,
);

testcase!(
    test_issubclass_tuple,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
def f(x: type[A | B | C]):
    if issubclass(x, (A, B)):
        assert_type(x, type[A] | type[B])
    else:
        assert_type(x, type[C])
    "#,
);

testcase!(
    test_isinstance_alias,
    r#"
from typing import assert_type
X = int
def f(x: str | int):
    if isinstance(x, X):
        assert_type(x, int)
    "#,
);

testcase!(
    test_isinstance_alias_of_union,
    r#"
class A: ...
class B(A): ...
class C(A): ...

X = B | C

def f(x: A) -> X:
    if isinstance(x, X):
        return x
    raise ValueError()
    "#,
);

// Using scoped type aliases with isinstance is a runtime error.
testcase!(
    test_isinstance_alias_error,
    r#"
type X = int
type Y = int | str
isinstance(1, X)  # E: Expected class object
isinstance(1, Y)  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_error,
    r#"
from typing import assert_type
def f(x: int | list[int]):
    if isinstance(x, list[int]):  # E: Expected class object
        # Either `int | list[int]` or `list[int]` is acceptable for the narrowed type.
        assert_type(x, list[int])
    "#,
);

testcase!(
    test_isinstance_parameterized_type,
    r#"
from typing import assert_type
def f(x: int | list[int], y: type[list[int]]):
    # Note that a literal `list[int]` as the second argument is illegal, but this is ok because
    # `y` may be a class object at runtime.
    if isinstance(x, y):
        assert_type(x, list[int])
    "#,
);

testcase!(
    bug = "We mistakenly think y[0] is a parameterized type because of the square brackets",
    test_isinstance_subscript_bug,
    r#"
def f(x, y: list[type[list[int]]]):
    return isinstance(x, y[0])  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_aliased,
    r#"
from typing import assert_type
istype = isinstance
def f(x: int | str):
    if istype(x, int):
        assert_type(x, int)
    "#,
);

testcase!(
    test_guarded_attribute_access_and,
    r#"
class A:
    x: str
class B:
    pass
def f(x: A | B):
    return isinstance(x, A) and x.x
    "#,
);

testcase!(
    test_guarded_attribute_access_or,
    r#"
class A:
    x: str
def f(x: A | None):
    return x is None or x.x
    "#,
);

testcase!(
    test_and_chain_with_walrus,
    r#"
from typing import assert_type, Literal

class A: ...
class B: ...

def test(x: A | B):
    y = isinstance(x, A) and (z := True)
    assert_type(x, A | B)
    assert_type(z, Literal[True])
    "#,
);

testcase!(
    test_typeguard_basic,
    r#"
from typing import TypeGuard, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_black_cat(x: Cat | Dog) -> TypeGuard[Cat]:
    return isinstance(x, Cat) and x.color == "black"
def f(x: Cat | Dog):
    if is_black_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Cat | Dog)
    is_black_cat(1)  # E: Argument `Literal[1]` is not assignable to parameter `x` with type `Cat | Dog` in function `is_black_cat`
    "#,
);

testcase!(
    test_typeis,
    r#"
from typing import TypeIs, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_cat(x: Cat | Dog) -> TypeIs[Cat]:
    return isinstance(x, Cat)
def f(x: Cat | Dog):
    if is_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Dog)
    "#,
);

testcase!(
    test_typeis_union,
    r#"
from typing import TypeIs, assert_type
class A: ...
class B: ...
class C: ...
def is_a_or_b(x: object) -> TypeIs[A | B]:
    return isinstance(x, A) or isinstance(x, B)
def f(x:  A | B | C, y: A | C):
    if is_a_or_b(x):
        assert_type(x, A | B)
    else:
        assert_type(x, C)
    if is_a_or_b(y):
        assert_type(y, A)
    else:
        assert_type(y, C)
    "#,
);

testcase!(
    test_narrow_and,
    r#"
from typing import assert_type
foo: dict[str, str] = {}
if "foo" in foo and foo["foo"] is not "as":
    val = foo["foo"]
    assert_type(val, str)
"#,
);

testcase!(
    test_issubclass,
    r#"
from typing import assert_type
class A: ...
class B(A): ...
def f(x: type[B] | type[int]):
    if issubclass(x, A):
        assert_type(x, type[B])
    else:
        assert_type(x, type[int])
    "#,
);

testcase!(
    test_issubclass_error,
    r#"
def f(x: int):
    if issubclass(x, int):  # E: Argument `int` is not assignable to parameter `cls` with type `type`
        return True
    "#,
);

testcase!(
    test_typeguard_instance_method,
    r#"
from typing import TypeGuard, assert_type
class C:
    def is_positive_int(self, x: object) -> TypeGuard[int]:
        return isinstance(x, int) and x > 0
def f(c: C, x: int | str):
    if c.is_positive_int(x):
        assert_type(x, int)
    "#,
);

testcase!(
    test_typeguard_generic_function,
    r#"
from typing import TypeGuard, assert_type
def f[T](x: object, y: T, z: T) -> TypeGuard[int]: ...
def f2[T](x: object, y: T) -> TypeGuard[T]: ...
def g(x: int | str):
    if f(x, 0, 0):
        assert_type(x, int)
    if f2(x, ""):
        assert_type(x, str)
    "#,
);

testcase!(
    test_implicit_else,
    r#"
from typing import assert_type
def f(x: int | None):
    if not x:
        return
    assert_type(x, int)
    "#,
);

testcase!(
    test_narrowed_elif_test,
    r#"
def f(x: int | None, y: bool):
    if not x:
        pass
    elif x > 42:
        pass
"#,
);

testcase!(
    test_narrow_comprehension,
    r#"
from typing import assert_type
def f(xs: list[int | None]):
    ys = [x for x in xs if x]
    assert_type(ys, list[int])
"#,
);

// Note: the narrowing code isn't actually what's giving us this behavior,
// it comes from flow-aware type information taking precedence over static
// annotations. But the end result is narrowing behavior.
testcase!(
    test_assignment_and_narrowing,
    r#"
from typing import assert_type, Literal
def foo(x: int | str):
    y: int | str = x
    assert_type(x, int | str)
    assert_type(y, int | str)
    x = 42
    y = 42
    assert_type(x, Literal[42])
    assert_type(y, Literal[42])
    "#,
);

testcase!(
    test_bad_typeguard_return,
    r#"
from typing import TypeGuard
def f(x) -> TypeGuard[str]:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to expected return type `bool` of type guard functions
def g(x) -> TypeGuard[str]:  # E: Function declared to return `TypeGuard[str]` but is missing an explicit `return`
    pass
    "#,
);

testcase!(
    test_isinstance_any_second,
    r#"
from typing import Any
def f(x: int | str, y: Any):
    if isinstance(x, y):
        pass
    "#,
);

testcase!(
    test_isinstance_any_literally,
    r#"
from typing import Any
def f(x: int | str):
    if isinstance(x, Any): # E: Expected class object, got `type[Any]`
        pass
    "#,
);

testcase!(
    test_isinstance_any_first,
    r#"
from typing import Any, assert_type
def f(x: Any):
    if isinstance(x, bool):
        assert_type(x, bool)
    else:
        assert_type(x, Any)
"#,
);

testcase!(
    test_listcomp_if_control_flow,
    r#"
class C: pass
class D(C): pass
def accepts_d(x: D) -> None: pass
def f(x: list[C], z: C):
    if accepts_d(z) and isinstance(z, D):  # E: Argument `C` is not assignable to parameter `x` with type `D`
        pass
    [y for y in x if (accepts_d(y) and isinstance(y, D))]  # E: Argument `C` is not assignable to parameter `x` with type `D` in function `accepts_d`
    [None for y in x if C.error]  # E: Class `C` has no class attribute `error`
    "#,
);

testcase!(
    test_unittest_assert,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_true(self) -> None:
        x = foo()
        self.assertTrue(x is not None)
        assert_type(x, int)
    
    def test_false(self) -> None:
        x = foo()
        self.assertFalse(x is None)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_none,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_none(self) -> None:
        x = foo()
        self.assertIsNone(x)
        assert_type(x, None)
    
    def test_is_not_none(self) -> None:
        x = foo()
        self.assertIsNotNone(x)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_isinstance,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_instance(self) -> None:
        x = foo()
        self.assertIsInstance(x, int)
        assert_type(x, int)
    
    def test_is_not_instance(self) -> None:
        x = foo()
        self.assertNotIsInstance(x, int)
        assert_type(x, None)
"#,
);

testcase!(
    test_unittest_assert_equal,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> Literal[0, 1]: ...
class MyTest(TestCase):
    def test_equal(self) -> None:
        x = foo()
        self.assertEqual(x, 0)
        assert_type(x, Literal[0])
    
    def test_not_equal(self) -> None:
        x = foo()
        self.assertNotEqual(x, 0)
        assert_type(x, Literal[1])
"#,
);

testcase!(
    test_unittest_assert_is,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> bool: ...
class MyTest(TestCase):
    def test_is(self) -> None:
        x = foo()
        self.assertIs(x, True)
        assert_type(x, Literal[True])
    
    def test_is_not(self) -> None:
        x = foo()
        self.assertIsNot(x, True)
        assert_type(x, Literal[False])
"#,
);

testcase!(
    test_unittest_assert_in,
    r#"
from typing import assert_type, Literal
from unittest import TestCase
def foo() -> Literal[1, 2, 3]: ...
class MyTest(TestCase):
    def test_in(self) -> None:
        x = foo()
        self.assertIn(x, [1, 2])
        assert_type(x, Literal[1, 2])
    
    def test_not_in(self) -> None:
        x = foo()
        self.assertNotIn(x, [1, 2])
        assert_type(x, Literal[3])
"#,
);

// Make sure we catch illegal arguments to isinstance and issubclass even when we aren't narrowing.
testcase!(
    test_validate_class_object_no_narrow,
    r#"
def f(x):
    return isinstance(x, list[int])  # E: Expected class object
def g(x):
    return issubclass(x, list[int])  # E: Expected class object
    "#,
);

testcase!(
    test_isinstance_type_typevar,
    r#"
from typing import assert_type
def f[T](x, y: type[T]) -> T:
    if isinstance(x, y):
        return x
    raise ValueError()
    "#,
);

testcase!(
    test_isinstance_type_self,
    r#"
from typing import Self, TypeGuard
class A:
    def f(self, x) -> TypeGuard[Self]:
        return isinstance(x, type(self))
    "#,
);

testcase!(
    test_or_negation,
    r#"
from typing import assert_type
def f(x: int | None, y: int | None):
    if x is None or y is None:
        pass
    else:
        assert_type(x, int)
        assert_type(y, int)
"#,
);

testcase!(
    test_narrow_to_anonymous_intersection,
    r#"
from typing import assert_type
class A: pass
class B: pass
class C(A, B): pass  # not used, but demonstrates why the narrow is not Never
def f(x: A):
    if isinstance(x, B):
        # In theory we could use `A & B` here; any common sublcass like `C` is possible.
        # Given that we don't have intersections, we follow Pyre's lead and use `B`
        assert_type(x, B)
"#,
);

testcase!(
    test_nested_or_with_multiple_vars,
    r#"
from typing import assert_type
class Foo: pass
class Bar(Foo): pass
def f(x: object, y: object) -> None:
    if isinstance(x, Bar) or (isinstance(y, Foo) and isinstance(x, Foo)):
        assert_type(x, Bar | Foo)
        assert_type(y, Foo | object)
    else:
        assert_type(x, object)
        assert_type(y, object)
"#,
);

testcase!(
    test_narrow_in,
    r#"
from typing import Literal, assert_type, Never
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
def test(x: Literal["foo", 1] | Color | bool | None, y: object, z: Literal["f", "g"]) -> None:
    if x in (1, Color.RED, True):
        assert_type(x, Literal[1, Color.RED, True])
    else:
        assert_type(x, Literal["foo", Color.BLUE, Color.GREEN, False] | None)
    if x in [1, 2, 3, 4]:
        assert_type(x, Literal[1])
    if x in [2, 3, 4]:
        assert_type(x, Never)
    if x in [y, 1]:
        # we only narrow if the list only contains literals
        assert_type(x, Literal["foo", 1, Color.BLUE, Color.GREEN, Color.RED] | bool | None)
    if z in "foo":
        # we only narrow if the RHS is a list, set, tuple literal
        assert_type(z, Literal["f", "g"])
    if y in {1, Color.RED, True}:
        assert_type(y, Literal[1, Color.RED, True])
    else:
        assert_type(y, object)
"#,
);

testcase!(
    test_narrow_len,
    r#"
from typing import assert_type, Never, NamedTuple
class NT(NamedTuple):
    x: int
    y: int
def test(x: tuple[int, int], y: tuple[int, *tuple[int, ...], int], z: tuple[int, ...], nt: NT) -> None:
    if len(x) == 2:
        assert_type(x, tuple[int, int])
    else:
        assert_type(x, Never)
    if len(x) == 1:
        assert_type(x, Never)
    else:
        assert_type(x, tuple[int, int])
    if len(x) < 1:
        # we currently do not narrow lengths for anything besides == and !=
        assert_type(x, tuple[int, int])
    if len(x) != 1:
        assert_type(x, tuple[int, int])
    if len(x) == x[0]:
        # only narrow if RHS is a literal
        assert_type(x, tuple[int, int])
    if len(y) == 2:
        assert_type(y, tuple[int, int])
    else:
        assert_type(y, tuple[int, *tuple[int, ...], int])
    if len(y) == 3:
        assert_type(y, tuple[int, int, int])
    if len(y) == 1:
        # this can never be true, since y has 2 concrete elements in the prefix/suffix
        assert_type(y, Never)
    if len(z) == 1:
        assert_type(z, tuple[int])
    else:
        assert_type(z, tuple[int, ...])
    if len(z) == 3:
        assert_type(z, tuple[int, int, int])
    if len(z) == 2 or len(z) == 3:
        assert_type(z, tuple[int, int] | tuple[int, int, int])
    if len(nt) == 2:
        assert_type(nt, NT)
    else:
        assert_type(nt, Never)
    if len(nt) == 1:
        assert_type(nt, Never)
    else:
        assert_type(nt, NT)
"#,
);
