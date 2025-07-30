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
    reveal_type(C.foo)  # E: revealed type: BoundMethod[type[C], (cls: type[Self@C]) -> int]
    reveal_type(c.foo)  # E: revealed type: BoundMethod[type[C], (cls: type[Self@C]) -> int]
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
    reveal_type(C.foo)  # E: revealed type: (self: Self@C) -> int
    "#,
);

testcase!(
    test_abstract_property,
    r#"
from typing import assert_type
from abc import ABC, abstractproperty
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
    reveal_type(C.foo)  # E: revealed type: (self: Self@C, value: str)
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
    bug = "TODO(stroxler): type inference causes us to complain on the decorator application",
    test_class_property_descriptor,
    r#"
from typing import assert_type, Callable, Any
class classproperty[T, R]:
    def __init__(self, fget: Callable[[type[T]], R]) -> None: ...
    def __get__(self, obj: object, obj_cls_type: type[T]) -> R: ...
class C:
    @classproperty    # E: Argument `(cls: Self@C) -> int` is not assignable to parameter `fget` with type `(type[@_]) -> @_`
    def cp(cls) -> int:
        return 42
assert_type(C.cp, Any) # Should probably be int
assert_type(C().cp, Any) # Should probably be int
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
