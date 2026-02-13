/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_staticmethod_with_explicit_parameter_type,
    r#"
from typing import assert_type, reveal_type, Callable
class C:
    @staticmethod
    def foo() -> int:
        return 42
    @staticmethod
    def bar(x: int) -> int:
        return x
def f(c: C):
    assert_type(C.foo, Callable[[], int])
    assert_type(c.foo, Callable[[], int])
    reveal_type(C.bar)  # E: (x: int) -> int
    reveal_type(c.bar)  # E: (x: int) -> int
    assert_type(C.foo(), int)
    assert_type(c.foo(), int)
    assert_type(C.bar(42), int)
    assert_type(c.bar(42), int)
    "#,
);

testcase!(
    test_staticmethod_calls_with_implicit_parameter_type,
    r#"
from typing import assert_type, Callable, Any
class C:
    @staticmethod
    def bar(x) -> int:
        return 42
def f(c: C):
    assert_type(c.bar(42), int)
    assert_type(c.bar(42), int)
    "#,
);

testcase!(
    test_classmethod_access,
    r#"
from typing import reveal_type
class C:
    @classmethod
    def foo(cls) -> int:
        return 42
def f(c: C):
    reveal_type(C.foo)  # E: revealed type: (cls: type[C]) -> int
    reveal_type(c.foo)  # E: revealed type: (cls: type[C]) -> int
    "#,
);

testcase!(
    test_classmethod_calls_with_explicit_parameter_type,
    r#"
from typing import assert_type
class C:
    @classmethod
    def foo(cls: type[C]) -> int:
        return 42
def f(c: C):
    assert_type(C.foo(), int)
    assert_type(c.foo(), int)
    "#,
);

testcase!(
    test_classmethod_calls_with_implicit_parameter_type,
    r#"
from typing import assert_type
class C:
    @classmethod
    def foo(cls) -> int:
        return 42
def f(c: C):
    assert_type(C.foo(), int)
    assert_type(c.foo(), int)
    "#,
);

testcase!(
    test_read_only_property,
    r#"
from typing import assert_type, reveal_type
class C:
    @property
    def foo(self) -> int:
        return 42
def f(c: C):
    assert_type(c.foo, int)
    c.foo = 42  # E: Attribute `foo` of class `C` is a read-only property and cannot be set
    reveal_type(C.foo)  # E: revealed type: (self: C) -> int
    "#,
);

testcase!(
    test_abstract_property,
    r#"
from typing import assert_type
from abc import ABC, abstractproperty # E: `abstractproperty` is deprecated
class C(ABC):
    @abstractproperty
    def foo(self) -> int:
        return 42
def f(c: C):
    assert_type(c.foo, int)
    "#,
);

testcase!(
    test_property_with_setter,
    r#"
from typing import assert_type, reveal_type
class C:
    @property
    def foo(self) -> int:
        return 42
    @foo.setter
    def foo(self, value: str) -> None:
        pass
def f(c: C):
    assert_type(c.foo, int)
    c.foo = "42"
    reveal_type(C.foo)  # E: revealed type: (self: C, value: str)
    "#,
);

testcase!(
    test_property_with_setter_and_deleter,
    r#"
from typing import assert_type, reveal_type

class C:
    @property
    def foo(self) -> int:
        return 42

    @foo.setter
    def foo(self, value: int) -> None:
        pass

    @foo.deleter
    def foo(self) -> None:
        pass

def f(c: C) -> None:
    assert_type(c.foo, int)
    c.foo = 1
    reveal_type(C.foo)  # E: revealed type: (self: C, value: int)
    del c.foo
    "#,
);

testcase!(
    test_cached_property_assignment_allowed,
    r#"
from functools import cached_property
from typing import assert_type

class C:
    @cached_property
    def foo(self) -> int:
        return 42

def f(c: C) -> None:
    assert_type(c.foo, int)
    c.foo = 42
    "#,
);

testcase!(
    bug = "cached_property's __name__ should not exist and attrname should be a str",
    test_cached_property_attrname,
    r#"
from functools import cached_property
from typing import reveal_type

class C:
    @cached_property
    def foo(self) -> int:
        return 42

reveal_type(C.foo.__name__)  # E: revealed type: str
reveal_type(C.foo.attrname)  # E: revealed type: Any
    "#,
);

// Make sure we don't crash.
testcase!(
    test_staticmethod_class,
    r#"
@staticmethod
class C:
    pass
    "#,
);

testcase!(
    test_simple_user_defined_get_descriptor,
    r#"
from typing import assert_type
class D:
    def __get__(self, obj, classobj) -> int: ...
class C:
    d = D()
assert_type(C.d, int)
assert_type(C().d, int)
C.d = 42  # E: `Literal[42]` is not assignable to attribute `d` with type `D`
C().d = 42  # E:  Attribute `d` of class `C` is a read-only descriptor with no `__set__` and cannot be set
    "#,
);

testcase!(
    test_descriptor_dunder_call,
    r#"
from typing import assert_type
class SomeCallable:
    def __call__(self, x: int) -> str:
        return "a"
class Descriptor:
    def __get__(self, instance: object, owner: type | None = None) -> SomeCallable:
        return SomeCallable()
class B:
    __call__: Descriptor = Descriptor()
b_instance = B()
assert_type(b_instance(1), str)
    "#,
);

// Test that a descriptor-based __call__ returning the same class doesn't cause
// infinite recursion when called through a type variable bound. This pattern
// would also cause RecursionError at runtime, so the "not callable" error is correct.
testcase!(
    test_descriptor_dunder_call_self_referencing_via_typevar,
    r#"
from typing import TypeVar
class SelfDescriptor:
    def __get__(self, instance: object, owner: type | None = None) -> "SelfCallable":
        return SelfCallable()
class SelfCallable:
    __call__: SelfDescriptor = SelfDescriptor()
T = TypeVar("T", bound=SelfCallable)
def f(x: T) -> None:
    x()  # E: Expected a callable, got `T`
    "#,
);

// Test that instance-only attributes with descriptor types are not treated as descriptors.
// Descriptor protocol only applies to class-body initialized attributes; both annotation-only
// and method-initialized attributes should allow assignment.
testcase!(
    test_instance_only_attribute_does_not_have_descriptor_semantics,
    r#"
from typing import assert_type

class Device:
    def __get__(self, obj, classobj) -> int: ...

class AnnotationOnly:
    device: Device

class MethodInitialized:
    device: Device
    def __init__(self) -> None:
        self.device = Device()

def f(a: AnnotationOnly, m: MethodInitialized) -> None:
    # Writes should be allowed (not treated as read-only descriptor)
    a.device = Device()  # OK: annotation-only, not a descriptor
    m.device = Device()  # OK: method-initialized, not a descriptor
    # Reads should return Device, not int (descriptor __get__ not invoked)
    assert_type(a.device, Device)
    assert_type(m.device, Device)
    "#,
);

// Test that ClassVar annotations with descriptor types have descriptor semantics
// even without initialization, since ClassVar implies class-level attribute.
testcase!(
    test_classvar_descriptor_without_initialization,
    r#"
from typing import ClassVar, assert_type

class ReadOnlyDescriptor:
    def __get__(self, obj, classobj) -> int: ...

# ClassVar implies class-level attribute, so descriptor semantics apply.
# Reading C.value invokes __get__ and returns int.
class C:
    value: ClassVar[ReadOnlyDescriptor]

def f() -> None:
    assert_type(C.value, int)
    "#,
);

// Test that annotation-only fields in child classes inherit parent descriptor behavior
// when the annotation type is compatible with the parent's descriptor type.
testcase!(
    test_annotation_only_child_inherits_parent_descriptor,
    r#"
from typing import assert_type

class ReadOnlyDescriptor:
    def __get__(self, obj, classobj) -> int: ...

class Parent:
    value: ReadOnlyDescriptor = ReadOnlyDescriptor()  # actual descriptor

# Child inherits parent's descriptor behavior since annotation type matches.
# Reading c.value invokes __get__ and returns int.
class Child(Parent):
    value: ReadOnlyDescriptor

def f(c: Child) -> None:
    assert_type(c.value, int)
    "#,
);

testcase!(
    test_simple_user_defined_set_descriptor,
    r#"
from typing import assert_type
class D:
    def __set__(self, obj, value: int) -> None: ...
class C:
    d = D()
assert_type(C.d, D)
assert_type(C().d, D)
C.d = 42  # E: `Literal[42]` is not assignable to attribute `d` with type `D`
C().d = 42
    "#,
);

testcase!(
    test_simple_user_defined_get_and_set_descriptor,
    r#"
from typing import assert_type
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class C:
    d = D()
assert_type(C.d, int)
assert_type(C().d, int)
C.d = "42"  # E: `Literal['42']` is not assignable to attribute `d` with type `D`
C().d = "42"
    "#,
);

testcase!(
    test_bound_method_preserves_function_attributes_from_descriptor,
    r#"
from __future__ import annotations

from typing import Callable


class CachedMethod:
    def __init__(self, fn: Callable[[Constraint], int]) -> None:
        self._fn = fn

    def __get__(self, obj: Constraint | None, owner: type[Constraint]) -> CachedMethod:
        return self

    def __call__(self, obj: Constraint) -> int:
        return self._fn(obj)

    def clear_cache(self, obj: Constraint) -> None: ...


def cache_on_self(fn: Callable[[Constraint], int]) -> CachedMethod:
    return CachedMethod(fn)


class Constraint:
    @cache_on_self
    def pointwise_read_writes(self) -> int:
        return 0

    def clear_cache(self) -> None:
        self.pointwise_read_writes.clear_cache(self)
    "#,
);

testcase!(
    test_class_property_descriptor,
    r#"
from typing import assert_type, Callable, Any
class classproperty[T, R]:
    def __init__(self, fget: Callable[[type[T]], R]) -> None: ...
    def __get__(self, obj: object, obj_cls_type: type[T]) -> R: ...
class C:
    @classproperty
    def cp(cls) -> int:
        return 42
assert_type(C.cp, int)
assert_type(C().cp, int)
C.cp = 42  # E: `Literal[42]` is not assignable to attribute `cp` with type `classproperty[C, int]`
C().cp = 42  # E:  Attribute `cp` of class `C` is a read-only descriptor with no `__set__` and cannot be set
    "#,
);

testcase!(
    test_generic_property,
    r#"
from typing import assert_type
class A:
    @property
    def x[T](self: T) -> T:
        return self
    @x.setter
    def x[T](self: T, value: T) -> None:
        pass
a = A()
assert_type(a.x, A)
a.x = a  # OK
a.x = 0  # E: `Literal[0]` is not assignable to parameter `value` with type `A`
    "#,
);

testcase!(
    test_property_attr,
    r#"
from typing import reveal_type
import types
class A:
    @property
    def f(self): return 0
reveal_type(A.f.fset)  # E: revealed type: ((Any, Any) -> None) | None
    "#,
);

testcase!(
    test_builtin_descriptors_on_awaitable_func,
    r#"
from typing import assert_type, Coroutine, Any
class A:
    async def f(self) -> int: return 0
    @classmethod
    async def g(cls) -> int: return 0
    @staticmethod
    async def h() -> int: return 0
def f(a: A):
    assert_type(a.f(), Coroutine[Any, Any, int])
    assert_type(A.g(), Coroutine[Any, Any, int])
    assert_type(A.h(), Coroutine[Any, Any, int])
    "#,
);

testcase!(
    test_descriptor_on_tvar_bound,
    r#"
from typing import assert_type
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class A:
    p = D()
def f[T: A](x: T):
    x.p = "foo"
    assert_type(x.p, int)
    "#,
);

testcase!(
    test_inherit_annotated_descriptor,
    r#"
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class A:
    d: D = D()
    def f(self):
        self.d = "ok"
class B(A):
    def f(self):
        self.d = "ok"
    "#,
);

testcase!(
    test_inherit_unannotated_descriptor,
    r#"
class D:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class A:
    d = D()
    def f(self):
        self.d = "ok"
class B(A):
    def f(self):
        self.d = "ok"
    "#,
);

// Regression test: at one point we were checking the raw class fields to
// see if something is a descriptor, which missed inherited behavior.
testcase!(
    test_descriptors_that_inherit,
    r#"
class DBase:
    def __get__(self, obj, classobj) -> int: ...
    def __set__(self, obj, value: str) -> None: ...
class D(DBase):
    pass
class A:
    d = D()
    def f(self):
        self.d = "ok"
    def g(self) -> int:
        return self.d
    "#,
);

testcase!(
    test_set_descriptor_on_class,
    r#"
from typing import overload

class D:
    @overload
    def __get__(self, obj: None, classobj: type) -> "D": ...
    @overload
    def __get__(self, obj: object, classobj: type) -> int: ...
    def __get__(self, obj: object | None, classobj: type) -> "D | int":
        if obj is None:
            return self
        return 42

    def __set__(self, obj: object, value: int) -> None: ...

class C:
    d: D = D()

    @classmethod
    def reset(cls) -> None:
        # Setting a descriptor on a class object (not an instance) should be
        # allowed because __set__ only intercepts instance assignments. Class
        # assignments bypass the descriptor protocol and write directly to
        # the class __dict__.
        cls.d = D()

# Static context: setting descriptor on class should also be allowed
C.d = D()

# Wrong type should still error as a type mismatch
C.d = "wrong"  # E: `Literal['wrong']` is not assignable to attribute `d` with type `D`
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1792
testcase!(
    test_descriptor_in_dataclass_transform,
    r#"
from typing import Any, dataclass_transform

class Mapped[T]:
    def __get__(self, obj, classobj) -> T: ...
    def __set__(self, obj, value: T) -> None: ...

def mapped_column(*args: Any, **kw: Any) -> Any: ...

@dataclass_transform(
    field_specifiers=(mapped_column,),
)
class DCTransformDeclarative(type):
    """metaclass that includes @dataclass_transforms"""

class MappedAsDataclass(metaclass=DCTransformDeclarative):
    pass

class DatasetMetadata(MappedAsDataclass):
    id: Mapped[str] = mapped_column(init=False)

DatasetMetadata()
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1803
testcase!(
    test_set_instance_attribute,
    r#"
from typing import assert_type

class MyDescriptor:
    def __get__(self, instance, owner=None):
        return 42

class A:
    def __init__(self):
        self.a = MyDescriptor()

assert_type(A().a, MyDescriptor)
    "#,
);

fn sqlalchemy_mapped_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "sqlalchemy.orm.base",
        r#"
class Mapped[T]:
    def __get__(self, instance, owner) -> T: ...
    def __set__(self, instance, value: T) -> None: ...
    def __delete__(self, instance) -> None: ...
    "#,
    );
    env.add_with_path(
        "sqlalchemy.orm.decl_api",
        "sqlalchemy/orm/decl_api.py",
        "class DeclarativeBase: ...",
    );
    env.add_with_path(
        "sqlalchemy.orm",
        "sqlalchemy/orm/__init__.py",
        r#"
from .base import Mapped as Mapped
from .decl_api import DeclarativeBase as DeclarativeBase
    "#,
    );
    env.add_with_path("sqlalchemy", "sqlalchemy/__init__.py", "");
    env
}

testcase!(
    test_sqlalchemy_mapped_is_always_descriptor,
    sqlalchemy_mapped_env(),
    r#"
from sqlalchemy.orm import DeclarativeBase, Mapped
class Base(DeclarativeBase):
    pass
class User(Base):
    name: Mapped[str]
    def __init__(self, name: str):
        self.name = name
    "#,
);
