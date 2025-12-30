/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_override_any,
    r#"
from typing import override, Any

class ParentB(Any):
    pass


class ChildB(ParentB):
    @override
    def method1(self) -> None:
        pass
 "#,
);

testcase!(
    test_override_basic_method,
    r#"

class A:
    def f(self, x:str, y:str) -> str:
        return x + y

class B(A):
    def f(self, x:int, y:int) -> int: # E: Class member `B.f` overrides parent class `A` in an inconsistent manner
        return x + y
 "#,
);

testcase!(
    test_override_basic_field,
    r#"
class A:
    x: int
    y: bool
    z: bool

class B(A):
    pass

class C(B):
    x: int
    y: str # E: Class member `C.y` overrides parent class `B` in an inconsistent manner
 "#,
);

testcase!(
    test_override_class_var,
    r#"
from typing import ClassVar
class A:
    x: int = 1
class B:
    x: ClassVar[int] = 1
class C(A):
    x: ClassVar[int] = 1  # E: ClassVar `C.x` overrides instance variable of the same name in parent class `A`
class D(B):
    x: ClassVar[int] = 1  # OK
class E(B):
    x: int = 1  # E: Instance variable `E.x` overrides ClassVar of the same name in parent class `B`
 "#,
);

testcase!(
    test_override_final_var,
    r#"
from typing import Final
class A:
    x: Final = 1
    y: Final[int] = 1
class B(A):
    x = 1  # E: `x` is declared as final in parent class `A`
    y = 1  # E: `y` is declared as final in parent class `A`
 "#,
);

testcase!(
    test_overload_override,
    r#"
from typing import overload

class A:

    @overload
    def method(self, x: int) -> int:
        ...

    @overload
    def method(self, x: str) -> str:
        ...

    def method(self, x: int | str) -> int | str:
        return 0


class B(A):

    @overload
    def method(self, x: int) -> int:
        ...

    @overload
    def method(self, x: str) -> str:
        ...

    def method(self, x: int | str) -> int | str:
        return 0
 "#,
);

testcase!(
    test_override_generic_simple,
    r#"
class A:
    def m[T](self, x: T) -> T: ...

class B(A):
    def m[T](self, x: T) -> T: ...   # OK

class C(A):
    def m(self, x: int) -> int: ...  # E: `C.m` overrides parent class `A` in an inconsistent manner
    "#,
);

testcase!(
    test_override_generic_bounds,
    r#"
class A: ...
class B(A): ...
class C(B): ...

class Base:
    def m[T: B](self, x: T) -> T: ...

class Derived1(Base):
    def m[T: A](self, x: T) -> T: ...  # OK, [T: A] accepts all types that [T: B] does

class Derived2(Base):
    def m[T: C](self, x: T) -> T: ...  # E: `Derived2.m` overrides parent class `Base` in an inconsistent manner
    "#,
);

testcase!(
    test_override_infers_parameter_type,
    r#"
from typing import Any, Generic, TypeVar, assert_type, override

T = TypeVar("T")

class G(Generic[T]):
    def foo(self, value: T) -> T:
        return value

class C(G[int]):
    @override
    def foo(self, value):
        assert_type(value, int)
        return value

    def bar(self, value):
        assert_type(value, Any)
    "#,
);

testcase!(
    test_override_requires_decorator,
    r#"
from typing import Any, assert_type

class Base:
    def foo(self, x: int) -> None: ...

class Derived(Base):
    def foo(self, x):
        assert_type(x, Any)
    "#,
);

testcase!(
    test_override_static_and_class_methods,
    r#"
from typing import assert_type, override

class Base:
    @staticmethod
    def sm(x: int) -> None: ...

    @classmethod
    def cm(cls, y: str) -> None: ...

class Derived(Base):
    @override
    @staticmethod
    def sm(x):
        assert_type(x, int)

    @override
    @classmethod
    def cm(cls, y):
        assert_type(y, str)
    "#,
);

testcase!(
    test_override_parameter_kinds,
    r#"
from typing import assert_type, override

class Base:
    def g(self, /, x: int, *args: str, y: bool, **kwargs: float) -> None: ...

class Derived(Base):
    @override
    def g(self, /, x, *args, y, **kwargs):
        assert_type(x, int)
        assert_type(args, tuple[str, ...])
        assert_type(y, bool)
        assert_type(kwargs, dict[str, float])
    "#,
);

testcase!(
    test_decorator_hint_takes_precedence_over_parent_hint,
    r#"
from typing import Any, assert_type, Callable, override

def decorate(x: Callable[[Any, float], None]) -> Callable[[Any, float], None]:
    return x

class A:
    def f(self, x: int, /):
        pass

class B(A):
    @decorate
    @override
    def f(self, x, /):
        assert_type(x, float)
    "#,
);

testcase!(
    test_mixed_decorator_and_parent_hints,
    r#"
from typing import Any, assert_type, Callable, override

def decorate(x: Callable[[Any, float], None]) -> Any:
    return x

class A:
    def f(self, x: int, y: str | None = None, /):
        pass

class B(A):
    @decorate
    @override
    def f(self, x, y=None, /):
        assert_type(x, float)
        assert_type(y, str | None)
    "#,
);

testcase!(
    test_overloaded_parent_provides_no_hint,
    r#"
from typing import Any, assert_type, overload, override

class A:
    @overload
    def f(self, x: int) -> int: ...
    @overload
    def f(self, x: str) -> str: ...
    def f(self, x):
        return x

class B(A):
    @override
    def f(self, x):
        assert_type(x, Any)
        return x

    "#,
);

testcase!(
    test_no_base_override,
    r#"
from typing import override

class A:
    def method1(self) -> int:
        return 1


class B(A):
    @override
    def method2(self) -> int: # E: Class member `B.method2` is marked as an override, but no parent class has a matching attribute
        return 1
 "#,
);

testcase!(
    test_default_value_consistent,
    r#"
class A:
    x: int

class B(A):
    def __init__(self) -> None:
        self.x = 0
 "#,
);

testcase!(
    test_default_value_inconsistent,
    r#"
class A:
    x: str

class B(A):
    def __init__(self) -> None:
        self.x = 0 # E: `Literal[0]` is not assignable to attribute `x` with type `str`
 "#,
);

testcase!(
    test_override_decorators,
    r#"
from typing import override

class ParentA:
    pass

class ChildA(ParentA):
    @staticmethod
    @override
    def static_method1() -> int: # E: Class member `ChildA.static_method1` is marked as an override, but no parent class has a matching attribute
        return 1

    @classmethod
    @override
    def class_method1(cls) -> int: # E: Class member `ChildA.class_method1` is marked as an override, but no parent class has a matching attribute
        return 1

    @property
    @override
    def property1(self) -> int: # E: Class member `ChildA.property1` is marked as an override, but no parent class has a matching attribute
        return 1

 "#,
);

testcase!(
    test_override_decorators_switched,
    r#"
from typing import override

class ParentA:
    pass

class ChildA(ParentA):
    @override
    @staticmethod
    def static_method1() -> int: # E: Class member `ChildA.static_method1` is marked as an override, but no parent class has a matching attribute
        return 1

 "#,
);

testcase!(
    test_override_custom_wrapper,
    r#"
from typing import Any, Callable, override

def wrapper(func: Callable[..., Any], /) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    return wrapped


class ParentA:

    @staticmethod
    def static_method1() -> int:
        return 1

class ChildA(ParentA):

    @wrapper
    @override
    @staticmethod
    def static_method1() -> bool:
        return True
 "#,
);

testcase!(
    test_override_duplicate_decorator,
    r#"
from typing import  override

class ParentA:

    @staticmethod
    def static_method1() -> int:
        return 1

class ChildA(ParentA):

    @staticmethod
    @override
    @staticmethod
    def static_method1() -> int:
        return 1
 "#,
);

testcase!(
    test_overload_override_error,
    r#"

from typing import overload, override

class ParentA:
    ...

class ChildA(ParentA):
    @overload
    def method4(self, x: int) -> int:  # E: no parent class has a matching attribute
        ...

    @overload
    def method4(self, x: str) -> str:
        ...

    @override
    def method4(self, x: int | str) -> int | str:
        return 0
 "#,
);

testcase!(
    test_override_final_method,
    r#"
from typing import final

class Parent:
    @final
    def a(self): ...

class Child(Parent):
    def a(self): ...  # E: `a` is declared as final in parent class `Parent`
 "#,
);

testcase!(
    test_override_literal_attr,
    r#"
from typing import Literal
class A:
    X: Literal[0] = 0
class B(A):
    X = 0
    "#,
);

testcase!(
    test_bad_override_literal_attr,
    r#"
from typing import Literal
class A:
    x: Literal[0] = 0
class B(A):
    x = 1 # E: `Literal[1]` is not assignable to attribute `x` with type `Literal[0]`
    "#,
);

testcase!(
    test_override_context_manager,
    r#"
import contextlib
import abc

class Parent:
    @contextlib.asynccontextmanager
    @abc.abstractmethod
    async def run(self):
        yield

class Child(Parent):
    @contextlib.asynccontextmanager
    async def run(self):
        yield
    "#,
);

testcase!(
    test_override_with_args_and_kwargs,
    r#"
from typing import *

class A:
    def test1(self): ...
    def test2(self, x: int, /): ...
    def test3(self, x: int = 1, /): ...
    def test4(self, x: int): ...
    def test5(self, x: int = 1): ...

class B(A):
    def test1(self, *args: int, **kwargs: int): ...
    def test2(self, *args: int, **kwargs: int): ...
    def test3(self, *args: int, **kwargs: int): ...
    def test4(self, *args: int, **kwargs: int): ...
    "#,
);

testcase!(
    test_override_with_unannotated_args_and_kwargs,
    r#"
from typing import *

class A:
    def test1(self): ...
    def test2(self, x: int, /): ...
    def test3(self, x: int = 1, /): ...
    def test4(self, x: int): ...
    def test5(self, x: int = 1): ...

class B(A):
    def test1(self, *args, **kwargs): ...
    def test2(self, *args, **kwargs): ...
    def test3(self, *args, **kwargs): ...
    def test4(self, *args, **kwargs): ...
    "#,
);

testcase!(
    test_override_with_args_and_kwonly,
    r#"
from typing import *

class A:
    def test1(self, x: int, /): ...
    def test2(self, x: int = 1, /): ...
    def test3(self, x: int): ...
    def test4(self, x: int): ...

class B(A):
    def test1(self, *args: int): ...
    def test2(self, *args: int): ...
    # This one is not OK because the kw-only x is required
    # If A.test3 passes x positionally then it will crash
    def test3(self, *args: int, x: int): ...  # E: Class member `B.test3` overrides parent class `A` in an inconsistent manner
    def test4(self, *args: int, x: int = 1): ...
    "#,
);

testcase!(
    test_unannotated_descriptor_override_error,
    r#"
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class A:
    d = D()
class B(A):
    d = 42  # E: `B.d` and `A.d` must both be descriptors
    "#,
);

testcase!(
    test_annotated_descriptor_override_error,
    r#"
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class A:
    d: D = D()
class B(A):
    d: int = 42  # E: `B.d` and `A.d` must both be descriptors
    "#,
);

testcase!(
    test_inherit_type_attribute,
    r#"
class ParentAttr: ...
class ChildAttr(ParentAttr): ...

class A:
    Attr: type[ParentAttr]
class B(A):
    Attr = ChildAttr
    "#,
);

testcase!(
    test_staticmethod_can_override_staticmethod,
    r#"
class Base:
    @staticmethod
    def method() -> int:
        return 1

def a_method() -> int:
    return 1

class Derived(Base):
    method = staticmethod(a_method)
    "#,
);

testcase!(
    test_override_self,
    r#"
from typing import Self
class Base:
    def covariant(self) -> Self:
        return self

    def contravariant(self, other: Self) -> None:
        pass

    def invariant(self) -> list[Self]:
        return []

class Derived(Base):
    def covariant(self) -> Self:
        return self

    def contravariant(self, other: Self) -> None:
        pass

    def invariant(self) -> list[Self]:
        return []
    "#,
);

testcase!(
    test_param_name_change,
    r#"
class A:
    def f(self, x: int):
        pass
class B(A):
    def f(self, x1: int):  # E: Got parameter name `x1`, expected `x`
        pass
    "#,
);

testcase!(
    test_override_dunder_names,
    r#"
from typing import Iterator

class Base: pass
class Derived(Base): pass

class UseBase:
    __private: Base
    def __iter__(self) -> Iterator[list[Base]]: ...

class UseDerived(UseBase):
    __private: Derived
    def __iter__(self) -> Iterator[list[Derived]]: ...  # E: `UseDerived.__iter__` overrides parent class `UseBase` in an inconsistent manner
    "#,
);

testcase!(
    test_override_dunder_names_empty_base,
    r#"
from typing import override

# Make sure the override check is always carried out even when the base class list is empty
class C:
    @override
    def __eq__(self, other: object) -> bool:  # OK
        ...

    @override
    def __does_not_exist__(self, other: object) -> bool:  # E: no parent class has a matching attribute
        ...
    "#,
);

testcase!(
    bug = "We currently skip checking overrides of `__call__`, which is a soundness hole",
    test_override_dunder_call,
    r#"
class Base: pass
class Derived(Base): pass

class UseBase:
    def __call__(self) -> list[Base]: ...

class UseDerived(UseBase):
    def __call__(self) -> list[Derived]: ...
    "#,
);

testcase!(
    test_override_with_type_alias_param,
    r#"
from typing import TypeAliasType
TA1 = TypeAliasType("TA1", float | int | None)
TA2 = TypeAliasType("TA2", float | int)
class A:
    def f(self, x: TA1):
        pass
class B(A):
    def f(self, x: TA2):  # E: `B.f` has type `BoundMethod[B, (self: B, x: TA2) -> None]`, which is not assignable to `BoundMethod[B, (self: B, x: TA1) -> None]`, the type of `A.f`
        pass
    "#,
);

testcase!(
    test_unannotated_none_attribute_override,
    r#"
from typing import Any, assert_type

class Foo:
    x = None

class Bar(Foo):
    x = 1

assert_type(Bar.x, Any | None)
assert_type(Foo.x, Any | None)
def test(x: type[Foo]):
    assert_type(x.x, Any | None)
    "#,
);

testcase!(
    test_unannotated_none_attribute_override_multiple,
    r#"
class Parent:
    x = None
    y = None

class Child1(Parent):
    x = 42

class Child2(Parent):
    y = "hello"
    "#,
);

testcase!(
    test_unannotated_none_attribute_with_annotation_ok,
    r#"
from typing import Optional

class Foo:
    x: Optional[int] = None

class Bar(Foo):
    x = 1  # OK - explicit Optional annotation allows override
    "#,
);

// Test for https://github.com/facebook/pyrefly/issues/1372
testcase!(
    test_property_setter_inheritance,
    r#"
from typing import *

class Myclass:
    @property
    def hello(self) -> int:
        return 1

    @hello.setter
    def hello(self, value: int):
        pass

class MySubclass(Myclass):
    def __init__(self):
        self.hello = 1
"#,
);

testcase!(
    test_override_method_with_unknown_callable,
    r#"
from typing import Callable

class A:
    def f(self) -> None:
        pass

def decorate(f: Callable) -> Callable:
    return f

class B(A):
    @decorate
    def f(self) -> None:
        pass
    "#,
);

testcase!(
    test_override_staticmethod,
    r#"
class A:
    @staticmethod
    def f() -> A | None:
        return A()

class B(A):
    @staticmethod
    def f() -> B:
        return B()
    "#,
);

testcase!(
    test_override_instance_method_with_staticmethod,
    r#"
class A:
    def f(self) -> None:
        pass

class B(A):
    @staticmethod
    def f() -> None:
        pass
    "#,
);

testcase!(
    test_override_instance_method_with_staticmethod_2,
    r#"
class A:
    def f(self) -> A:
        return A()

class B(A):
    @staticmethod
    def f() -> B:
        return B()
    "#,
);

testcase!(
    test_annotated_union,
    r#"
from typing import assert_type
class A:
    x: int | None = 0
class B(A):
    x = 1
class C(A):
    x = "oops"  # E: `Literal['oops']` is not assignable to attribute `x` with type `int | None`
def f(b: B):
    assert_type(b.x, int | None)
    "#,
);

testcase!(
    test_inferred_union,
    r#"
from typing import assert_type
def f() -> int | None: ...
class A:
    x = f()
class B(A):
    x = 1
def f(b: B):
    assert_type(b.x, int | None)
    "#,
);

testcase!(
    test_violate_inherited_type,
    r#"
class A:
    x = 1
class B(A):
    x = "oops"  # E: `B.x` has type `str`, which is not consistent with `int`
    "#,
);

testcase!(
    test_use_inherited_type_as_context,
    r#"
from typing import assert_type

class A: ...
class B(A): ...
class C:
    x = [A()]
class D(C):
    x = [B()]

def f(d: D):
    assert_type(d.x, list[A])
    "#,
);

// Tests for MissingOverrideDecorator (strict override enforcement)
testcase!(
    test_missing_override_decorator,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    def foo(self, x: int) -> None: ...

class Derived(Base):
    def foo(self, x: int) -> None: ...  # E: Class member `Derived.foo` overrides a member in a parent class but is missing an `@override` decorator
    "#,
);

testcase!(
    test_missing_override_decorator_with_override,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
from typing import override

class Base:
    def foo(self, x: int) -> None: ...

class Derived(Base):
    @override
    def foo(self, x: int) -> None: ...  # OK - has @override
    "#,
);

testcase!(
    test_missing_override_decorator_no_parent_method,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    def foo(self, x: int) -> None: ...

class Derived(Base):
    def bar(self, x: int) -> None: ...  # OK - not overriding anything
    "#,
);

testcase!(
    test_missing_override_decorator_classmethod,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    @classmethod
    def foo(cls) -> None: ...

class Derived(Base):
    @classmethod
    def foo(cls) -> None: ...  # E: Class member `Derived.foo` overrides a member in a parent class but is missing an `@override` decorator
    "#,
);

testcase!(
    test_missing_override_decorator_staticmethod,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    @staticmethod
    def foo() -> None: ...

class Derived(Base):
    @staticmethod
    def foo() -> None: ...  # E: Class member `Derived.foo` overrides a member in a parent class but is missing an `@override` decorator
    "#,
);

testcase!(
    test_missing_override_decorator_property,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    @property
    def foo(self) -> int: ...

class Derived(Base):
    @property
    def foo(self) -> int: ...  # E: Class member `Derived.foo` overrides a member in a parent class but is missing an `@override` decorator
    "#,
);

testcase!(
    test_missing_override_decorator_field,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    x: int

class Derived(Base):
    x: int  # E: Class member `Derived.x` overrides a member in a parent class but is missing an `@override` decorator
    "#,
);

testcase!(
    test_missing_override_decorator_disabled_by_default,
    r#"
class Base:
    def foo(self, x: int) -> None: ...

class Derived(Base):
    def foo(self, x: int) -> None: ...  # OK when error is disabled by default
    "#,
);

testcase!(
    test_missing_override_decorator_multiple_inheritance,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base1:
    def foo(self) -> None: ...

class Base2:
    def bar(self) -> None: ...

class Derived(Base1, Base2):
    def foo(self) -> None: ...  # E: overrides a member in a parent class but is missing an `@override` decorator
    def bar(self) -> None: ...  # E: overrides a member in a parent class but is missing an `@override` decorator
    def baz(self) -> None: ...  # OK - not overriding anything
    "#,
);

testcase!(
    test_missing_override_decorator_dunder_method,
    TestEnv::new().enable_missing_override_decorator_error(),
    r#"
class Base:
    def __str__(self) -> str: ...  # E: overrides a member in a parent class but is missing an `@override` decorator

class Derived(Base):
    def __str__(self) -> str: ...  # E: overrides a member in a parent class but is missing an `@override` decorator
    "#,
);
