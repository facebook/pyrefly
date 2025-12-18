/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    reveal_type(C.foo)  # E: revealed type: BoundMethod[type[C], (cls: type[C]) -> int]
    reveal_type(c.foo)  # E: revealed type: BoundMethod[type[C], (cls: type[C]) -> int]
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
C.d = 42  # E: Attribute `d` of class `C` is a descriptor, which may not be overwritten
C().d = 42  # E:  Attribute `d` of class `C` is a read-only descriptor with no `__set__` and cannot be set
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
C.d = 42  # E: Attribute `d` of class `C` is a descriptor, which may not be overwritten
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
C.d = "42"  # E: Attribute `d` of class `C` is a descriptor, which may not be overwritten
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
C.cp = 42  # E: Attribute `cp` of class `C` is a descriptor, which may not be overwritten
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
