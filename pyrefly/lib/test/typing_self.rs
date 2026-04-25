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
    bug = "conformance: Should error when returning concrete class instead of Self",
    test_self_return_concrete_class,
    r#"
from typing import Self

class Shape:
    def method(self) -> Self:
        return Shape()  # should error: returns Shape, not Self

    @classmethod
    def cls_method(cls) -> Self:
        return Shape()  # should error: returns Shape, not Self
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

// Regression test for https://github.com/facebook/pyrefly/issues/3008
testcase!(
    test_self_outside_class_no_internal_error_via_classmethod_trampoline,
    r#"
from typing import Callable, Self, TypeVar

TReturn = TypeVar("TReturn")

def trampoline(orig: Callable[..., TReturn]) -> TReturn:
    ...

def person_create_with_name_trampoline(cls, *args, **kwargs):
    return trampoline(person_create_with_name_orig)

class Person:
    name: str
    create_with_name = classmethod(person_create_with_name_trampoline)

def person_create_with_name_orig(cls, name: str) -> Self:  # E: `Self` must appear within a class
    return cls()

Person.create_with_name("foo").name
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
    test_self_with_explicit_typevar_self,
    r#"
from typing import Self, TypeVar

TFoo = TypeVar("TFoo", bound="Foo")

class Foo:
    def ok_method(self) -> Self:
        raise NotImplementedError

    def bad_method(self: TFoo) -> Self:  # E: `Self` cannot be used when `self` has an explicit TypeVar annotation
        raise NotImplementedError

    def also_ok(self: TFoo) -> TFoo:
        raise NotImplementedError

    def bad_param(self: TFoo, other: Self) -> TFoo:  # E: `Self` cannot be used when `self` has an explicit TypeVar annotation
        raise NotImplementedError

TCls = TypeVar("TCls", bound=type["Foo"])

class Bar:
    @classmethod
    def bad_classmethod(cls: TCls) -> Self:  # E: `Self` cannot be used when `cls` has an explicit TypeVar annotation
        raise NotImplementedError

    @classmethod
    def ok_classmethod(cls) -> Self:
        raise NotImplementedError
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

    def __new__(cls, value: int) -> Self:
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
    bug = "Enum members should be assignable to Self",
    test_self_in_enum_classmethod,
    r#"
from typing import Self
from enum import Enum

class E(Enum):
    A = 1

    @classmethod
    def f(cls) -> Self:
        return cls.A  # E: Returned type `Literal[E.A]` is not assignable to declared return type `Self@E`
"#,
);

// Passing a concrete class to `cls.__new__` is incorrect when `cls` could be a subclass.
testcase!(
    bug = "Should error: concrete type[C] is not assignable to type[Self@C]",
    test_cls_new_with_concrete_class,
    r#"
class C:
    @classmethod
    def create(cls) -> C:
        return cls.__new__(C)
"#,
);

// Self is pinned when `[self]` is inferred, so `append(other: C)` fails against `Self@C`.
testcase!(
    bug = "Should error: C is not assignable to Self@C in list.append",
    test_self_in_container_pinning,
    r#"
class C:
    def foo(self, other: C) -> list:
        xs = [self]
        xs.append(other)
        return xs
"#,
);

// A type-preserving decorator (_Fn -> _Fn) should not cause Self to be lost in subclasses.
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

// Explicit class references in parameters (e.g. `other: Parent`) must not be converted to Self.
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

// Non-type-preserving decorators preserve Self when the decorator is generic.
testcase!(
    test_non_type_preserving_decorator_self_method,
    r#"
from typing import Self, Callable, TypeVar, Any, Concatenate

T = TypeVar("T")
R = TypeVar("R")
P = TypeVar("P")

def sig_copy(fn: Callable[Concatenate[T, ...], R]) -> Callable[[T, int], R]:
    return fn

class Parent:
    @sig_copy
    def method(self, **kwds: Any) -> Self:
        return self

class Child(Parent):
    def child_method(self) -> int:
        return 1

def test(c: Child) -> None:
    c.method(42).child_method()
"#,
);

// Regression test for a previously unhandled crash when computing intersection
// of SelfType and ClassType (after removing SelfType <: ClassType)
testcase!(
    test_classmethod_self_return_with_issubclass_narrowing,
    r#"
from typing import Self, assert_type

class Parent:
    @classmethod
    def decode(cls, data: dict) -> Self:
        if issubclass(cls, Child):
            # This narrowing creates an intersection of a Self type with a concrete type
            return cls(data, legacy=True)
        return cls(data)

    def __init__(self, data: dict) -> None:
        pass

class Child(Parent):
    def __init__(self, data: dict, legacy: bool = False) -> None:
        super().__init__(data)

assert_type(Parent.decode({}), Parent)
assert_type(Child.decode({}), Child)
"#,
);

// Regression test for inherited generic instantiations in ClassType & SelfType
// intersections. This exercises the soundness guard that only collapses
// `ClassType(Child) & SelfType(Parent[int])` when the upcast back to `Parent`
// preserves the same inherited instantiation.
testcase!(
    test_classmethod_self_return_with_generic_issubclass_narrowing,
    r#"
from typing import Self, assert_type

class Parent[T]:
    @classmethod
    def decode(cls) -> Self:
        if issubclass(cls, Child):
            return cls()
        return cls()

    def __init__(self) -> None:
        pass

class Child(Parent[int]):
    pass

assert_type(Child.decode(), Child)
"#,
);

// Regression test: descriptor access on `self` should preserve SelfType in the
// `owner` argument to `__get__`, so that a descriptor with a generic owner parameter
// doesn't produce false-positive errors.
testcase!(
    test_descriptor_preserves_self_type,
    r#"
from typing import Generic, TypeVar, final, Callable

T = TypeVar("T")
R = TypeVar("R")

@final
class classproperty(Generic[T, R]):
    def __init__(self, getter: Callable[[T], R]) -> None: ...
    def __get__(self, instance: object, owner: T) -> R: ...

class Base:
    @classproperty
    def name(cls) -> str:
        return "base"

    def use_name(self) -> str:
        return self.name

class Child(Base):
    pass

def test(c: Child) -> None:
    c.use_name()
"#,
);

// Regression test: descriptor setter on `self` should preserve SelfType in the
// `obj` argument to `__set__`.
testcase!(
    test_descriptor_setter_preserves_self_type,
    r#"
from typing import Generic, TypeVar, final

T = TypeVar("T")
V = TypeVar("V")

@final
class TypedField(Generic[T, V]):
    def __get__(self, instance: T, owner: type[T]) -> V: ...
    def __set__(self, instance: T, value: V) -> None: ...

class Base:
    field: TypedField[Base, str] = TypedField()

    def update(self) -> None:
        self.field = "hello"
"#,
);

testcase!(
    bug = "Should raise error in the overloads when returning concrete class instead of Self",
    test_overload_returning_self,
    r#"
from typing import Self, overload

class C:
    @overload
    def clone(self, x: int) -> C: ...
    @overload
    def clone(self, x: str) -> C: ...
    def clone(self, x) -> Self: ...
    "#,
);

testcase!(
    test_method_bad_receiver,
    r#"
class A: ...
class D:
    def __init__(self: A): pass  # E: `__init__` method self type `A` is not a superclass of class `D`
    def f(self: A): pass  # E: `f` method self type `A` is not a superclass of class `D`
    def g(self: type[A]): pass  # E: `g` method self type `type[A]` is not a superclass of class `D`
    def h(self: D): pass # Ok
    "#,
);

testcase!(
    test_method_type_of_defining_class_self,
    r#"
class D:
    def f(self: type[D]): pass # No error: type[D] where D is the defining class
    "#,
);

testcase!(
    test_overloaded_method_self_narrowing,
    r#"
from typing import overload
class A: ...
class D:
    @overload
    def f(self: A, x: int) -> int: ...  # No error: overload variants are not checked
    @overload
    def f(self, x: str) -> str: ...
    def f(self, x):
        return x
    "#,
);

testcase!(
    test_classmethod_bad_receiver,
    r#"
class A: ...
class D:
    @classmethod
    def f(cls: A): pass  # E: `f` method cls type `A` is not a valid `type[...]` annotation
    @classmethod
    def g(cls: type[A]): pass  # E: `g` method cls type `type[A]` is not a superclass of class `D`
    @classmethod
    def h(cls: type[D]): pass # Ok
"#,
);

testcase!(
    test_classmethod_bare_typevar_receiver,
    r#"
from typing import TypeVar
TCls = TypeVar("TCls", bound=type["D"])
class D:
    @classmethod
    def f(cls: TCls): pass # No error: bare TypeVar with type[X] bound is allowed
    "#,
);

testcase!(
    test_protocol_self_annotation,
    r#"
from typing import Protocol

class Proto(Protocol):
    def method(self) -> int: ...

class Impl:
    def method(self) -> int:
        return 0

    @classmethod
    def create(cls: type[Proto]) -> Proto:  # No error: Proto is a protocol
        return cls()

    def update(self: Proto) -> None:  # No error: Proto is a protocol
        pass
    "#,
);

testcase!(
    bug = "Should ideally check structural subtyping for protocol self/cls annotations",
    test_protocol_self_annotation_not_satisfied,
    r#"
from typing import Protocol

class Proto(Protocol):
    def method(self) -> int: ...

class NotImpl:
    @classmethod
    def create(cls: type[Proto]) -> Proto:  # No error, but NotImpl does not satisfy Proto
        return cls()

    def update(self: Proto) -> None:  # No error, but NotImpl does not satisfy Proto
        pass
    "#,
);
