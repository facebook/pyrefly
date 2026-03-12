/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_use_self,
    r#"
from typing import assert_type
from typing import Self
import typing
from typing import Self as Myself

class C:
    def m(self, x: Self, y: Myself) -> list[typing.Self]:
        return [self, x, y]

assert_type(C().m(C(), C()), list[C])
"#,
);

testcase!(
    test_typing_self_return,
    r#"
from typing import Self, assert_type
class A:
    def f(self) -> Self:
        return self
    @classmethod
    def g(cls) -> type[Self]:
        return cls
class B(A):
    pass
assert_type(B().f(), B)
assert_type(B().g(), type[B])
    "#,
);

testcase!(
    test_typing_self_param,
    r#"
from typing import Self
class A:
    def f(self, x: Self):
        pass
class B(A):
    pass
def f(a: A, b: B):
    b.f(b)  # OK
    b.f(a)  # E:
    "#,
);

testcase!(
    test_typing_self_new_param,
    r#"
from typing import Self
class A:
    def __new__(cls, x: type[Self]):
        return super().__new__(cls)
class B(A):
    pass
B(B)  # OK
B(A)  # E:
    "#,
);

testcase!(
    test_assert_type,
    r#"
from typing import Self, assert_type
class A:
    def __new__(cls, *args, **kwargs):
        assert_type(cls, type[Self])
        super().__new__(cls, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        assert_type(cls, type[Self])
        super().__init_subclass__(**kwargs)

    @classmethod
    def f1(cls):
        assert_type(cls, type[Self])

    def f2(self):
        assert_type(self, Self)

    def f3(self: Self) -> Self:
        assert_type(self, Self)
        return self
    "#,
);

testcase!(
    test_instance_attr,
    r#"
from typing import Self, assert_type
class A:
    x: Self
    y: int
    def f(self):
        assert_type(self.x, Self)
        assert_type(self.x.y, int)
class B(A):
    pass

assert_type(A().x, A)
assert_type(B().x, B)
    "#,
);

testcase!(
    test_class_attr,
    r#"
from typing import ClassVar, Self, assert_type
class A:
    x: ClassVar[Self]
    y: int
class B(A):
    pass

assert_type(A.x, A)
assert_type(B.x, B)
    "#,
);

testcase!(
    test_cast_self,
    r#"
from typing import cast, Self
class Foo:
    def hello(self): pass
    def check(self, other):
        other2 = cast(Self, other)
        other2.hello()
    "#,
);

testcase!(
    test_inherit_overloaded_dunder_new_with_self,
    r#"
from typing import Self, overload

class A:
    @overload
    def __new__(cls, x: int) -> Self: ...
    @overload
    def __new__(cls, x: str) -> Self: ...
    def __new__(cls, x: int | str) -> Self:
        return super().__new__(cls)

class B(A):
    pass

x: B = B(1)
"#,
);

testcase!(
    test_literal_self,
    r#"
from typing import Self, Literal, assert_type
import enum

class E(enum.Enum):
    A = 1
    B = 2

    def m(self, other: Self) -> Self:
        return other

a: Literal[E.A] = E.A

assert_type(a.m(E.B), E)
assert_type(E.A.m(E.B), E)
"#,
);

testcase!(
    test_callable_self,
    r#"
from typing import Self, assert_type

class C:
    def __call__(self) -> Self:
        return self

    def m(self):
        assert_type(self(), Self)
"#,
);

testcase!(
    test_self_return_concrete_class,
    r#"
from typing import Self

class Shape:
    def method(self) -> Self:
        return Shape()  # E: Returned type `Shape` is not assignable to declared return type `Self@Shape`

    @classmethod
    def cls_method(cls) -> Self:
        return Shape()  # E: Returned type `Shape` is not assignable to declared return type `Self@Shape`
"#,
);

testcase!(
    test_self_in_class_body_expression,
    r#"
from typing import Self, assert_type

class SomeClass:
    # Self in inferred class variable type
    cache = dict[int, Self]()

    def get_instance(self) -> Self:
        x = self.cache[0]
        if x:
            return x
        raise RuntimeError()

assert_type(SomeClass().cache, dict[int, SomeClass])
assert_type(SomeClass().get_instance(), SomeClass)
"#,
);

testcase!(
    test_self_outside_class,
    r#"
from typing import Self

def foo() -> Self: ... # E: `Self` must appear within a class
x: Self # E: `Self` must appear within a class
tupleSelf = tuple[Self] # E: `Self` must appear within a class
    "#,
);

testcase!(
    test_self_inside_class,
    r#"
from typing import Self

class A[T]: pass
class B(A[Self]): pass # E: `Self` must appear within a class
class C:
    @staticmethod
    def foo() -> Self: ... # E: `Self` cannot be used in a static method

    @staticmethod
    def bar(x: Self) -> None: ... # E: `Self` cannot be used in a static method

    @staticmethod
    def baz() -> list[Self]: ... # E: `Self` cannot be used in a static method
    "#,
);

testcase!(
    test_self_inside_metaclass,
    r#"
from typing import Self

class C(type):
    x: Self  # E: `Self` cannot be used in a metaclass
    def foo(cls) -> Self: ... # E: `Self` cannot be used in a metaclass
    def __new__(cls, x: Self) -> Self: ... # E: `Self` cannot be used in a metaclass  # E: `Self` cannot be used in a metaclass
    def __mul__(cls, count: int) -> list[Self]: ... # E: `Self` cannot be used in a metaclass
    "#,
);

testcase!(
    test_classmethod_cls_call_returns_self,
    r#"
from typing import Self

class Base:
    @classmethod
    def create(cls):
        return cls()

    @classmethod
    def create_annotated(cls) -> Self:
        return cls()

class Child(Base): pass

child1: Child = Child.create()  # OK - cls() returns Self
child2: Child = Child.create_annotated()  # OK with explicit Self annotation
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2526
testcase!(
    test_self_no_invalid_typevar,
    r#"
from typing import Generic, Self, TypeVar

X = TypeVar("X")
Y1 = TypeVar("Y1")
Y2 = TypeVar("Y2", default=Y1) # default value is important here

class A(Generic[X]):
    pass

class B(A[Y1 | Y2], Generic[Y1, Y2]):
    def __new__(cls, A: A[Y1], B: A[Y2]) -> Self: ...
    "#,
);

testcase!(
    test_self_return_in_classmethod,
    r#"
from typing import Self
class C:
    @classmethod
    def bar(cls) -> Self:
        return cls(1)

    def __new__(cls, value: int) -> Self:  # Accepted
        return cls(1)
"#,
);

testcase!(
    test_type_self_constructor_call,
    r#"
from typing import Self
class C:
    def foo(self) -> Self:
        return type(self)()
"#,
);

testcase!(
    bug = "Mypy allows this but Pyright doesn't, we should support it since enums are effectively final classes",
    test_self_in_enum_classmethod,
    r#"
from typing import Self
from enum import Enum

class E(Enum):
    A = 1

    @classmethod
    def f(cls) -> Self:
        return cls.A # E: Returned type `Literal[E.A]` is not assignable to declared return type `Self@E`
"#,
);

// This pattern exists in the open source codebase but is technically incorrect
testcase!(
    test_cls_new_with_concrete_class,
    r#"
class C:
    @classmethod
    def create(cls) -> C:
        return cls.__new__(C) # E: Argument `type[C]` is not assignable to parameter `cls` with type `type[Self@C]` in function `object.__new__`
"#,
);

// This is technically correct(?) but confusing to users, due to early pinning on the Self type.
testcase!(
    test_self_in_container_pinning,
    r#"
class C:
    def foo(self, other: C) -> list:
        xs = [self]
        xs.append(other) # E: Argument `C` is not assignable to parameter `object` with type `Self@C` in function `list.append`
        return xs
"#,
);

// A decorator that preserves the function signature (fn: _Fn -> _Fn)
// should not cause Self to be lost. When a subclass inherits the decorated method,
// the return type should still be Self (the subclass), not the parent class.
testcase!(
    test_decorated_self_method_in_subclass,
    r#"
from typing import Self, Callable, TypeVar, Any

_Fn = TypeVar("_Fn", bound=Callable[..., Any])
def deco(fn: _Fn) -> _Fn:
    return fn

class Parent:
    @deco
    def method(self) -> Self:
        return self

class Child(Parent):
    def child_method(self) -> int:
        return 1

def test(c: Child) -> None:
    c.method().child_method()
"#,
);

// When a decorated Self-returning method also has an explicit class reference in its
// parameters, the decorator should preserve Self in the return type without converting
// the explicit class reference to Self.
testcase!(
    test_decorated_self_method_with_explicit_class_param,
    r#"
from typing import Self, Callable, TypeVar, Any

_Fn = TypeVar("_Fn", bound=Callable[..., Any])
def deco(fn: _Fn) -> _Fn:
    return fn

class Parent:
    @deco
    def merge(self, other: Parent) -> Self:
        return self

class Child(Parent):
    def child_method(self) -> int:
        return 1

def test(c: Child, p: Parent) -> None:
    c.merge(p).child_method()
    c.merge(c).child_method()
"#,
);

// A non-type-preserving decorator that changes the parameters but preserves the return
// type variable still loses Self, because the overall function type changes.
// This mirrors altair's @use_signature pattern.
testcase!(
    bug = "Non-type-preserving decorator that keeps return type R still loses Self in R",
    test_non_type_preserving_decorator_self_method,
    r#"
from typing import Self, Callable, TypeVar, Any, Concatenate

T = TypeVar("T")
R = TypeVar("R")
P = TypeVar("P")

def sig_copy(fn: Callable[Concatenate[T, ...], R]) -> Callable[[T, int], R]:
    return fn  # type: ignore

class Parent:
    @sig_copy
    def method(self, **kwds: Any) -> Self:
        return self

class Child(Parent):
    def child_method(self) -> int:
        return 1

def test(c: Child) -> None:
    c.method(42).child_method()  # E: Object of class `Parent` has no attribute `child_method`
"#,
);
