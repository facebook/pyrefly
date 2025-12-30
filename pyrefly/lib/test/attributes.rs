/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

// Test case for various edge cases where a name isn't in the flow, and we might
// or might not decide an attribute has been defined.
testcase!(
    test_semantics_for_when_class_body_defines_attributes,
    r#"
from typing import assert_type, Any
def condition() -> bool: ...
class A:
    # Annotated with no value
    b: int
    # Annotated twice with no value
    c: str
    c: str
    # Defined in conditional control flow, with and without annotation
    if condition():
        d = 42
        e: int = 42
    # Defined (with or without annotation) but only in terminating control flow
    if condition():
        f = 42
        g: int = 42
        exit()
    h = 42
    i: int = 42
    del h
    del i
assert_type(A.b, int)
assert_type(A.c, str)
assert_type(A.d, int)
assert_type(A.e, int)
assert_type(A.f, Any)  # E: Class `A` has no class attribute `f`
assert_type(A.g, Any)  # E: Class `A` has no class attribute `g`
assert_type(A.h, int)
assert_type(A.i, int)
    "#,
);

testcase!(
    test_set_attribute,
    r#"
class A:
    x: int
def f(a: A):
    a.x = 1  # OK
    a.x = "oops"  # E: `Literal['oops']` is not assignable to attribute `x` with type `int`
    "#,
);

testcase!(
    test_set_attribute_in_unpacked_assign,
    r#"
class A:
    x: int
    y: str
def f(a: A):
    a.x, a.y = "x", "y"  # E: `Literal['x']` is not assignable to attribute `x` with type `int`
    "#,
);

testcase!(
    test_self_attribute_unannotated,
    r#"
from typing import assert_type
class A:
    def __init__(self, x: int):
        self.x = x
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_unannotated_attribute_bad_assignment,
    r#"
class A:
    def __init__(self):
        self.x = 0
    def f(self):
        self.x = "oops"  # E: `Literal['oops']` is not assignable to attribute `x` with type `int`
    "#,
);

testcase!(
    test_super_object_bad_assignment,
    r#"
class A:
    a: int = 3

class B(A): pass

super(B, B()).a = 3  # E: Cannot set field `a`
    "#,
);

testcase!(
    test_super_object_delete_error,
    r#"
class A:
    a: int = 3

class B(A): pass
del super(B, B()).a # E: Cannot delete field `a`
    "#,
);

testcase!(
    test_super_retain_self,
    r#"
from typing import Self
class A:
    def m(self) -> Self:
        return self

    @classmethod
    def classm(cls) -> type[Self]:
        return cls

class B(A):
    def m(self) -> Self:
        return super().m()

    @classmethod
    def classm(cls) -> type[Self]:
        return super().classm()
    "#,
);

testcase!(
    test_self_attribute_assign_twice,
    r#"
from typing import assert_type
class A:
    def f(self, x: str):
        self.x = x  # E: `str` is not assignable to attribute `x` with type `int`
    def __init__(self, x: int):
        self.x = x
    "#,
);

testcase!(
    test_self_attribute_in_unrecognized_method_enabled,
    TestEnv::new().enable_implicitly_defined_attribute_error(),
    r#"
from typing import assert_type
class A:
    def f(self, x: int):
        self.x = x  # E: Attribute `x` is implicitly defined by assignment in method `f`, which is not a constructor
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_self_attribute_in_unrecognized_method_default_disabled,
    r#"
from typing import assert_type
class A:
    def f(self, x: int):
        self.x = x
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_inherited_attribute_in_unrecognized_method,
    r#"
from typing import assert_type
class A:
    x: int
class B(A):
    def f(self, x: int):
        self.x = x
    "#,
);

// Verify that we correctly pick up parant class type as context when there's a
// qualifier-only annotation.
testcase!(
    test_inherited_attribute_with_qualifier_only_annotation,
    r#"
from typing import ClassVar, assert_type
class A: pass
class B(A): pass
class Foo:
    x: ClassVar[list[A]] = []
    y: ClassVar[list[A]] = []
class Bar(Foo):
    x: ClassVar = [B()]
    y = [B()]
assert_type(Bar.x, list[A])
assert_type(Bar.y, list[A])
    "#,
);

// Ref https://github.com/facebook/pyrefly/issues/370
// Ref https://github.com/facebook/pyrefly/issues/522
testcase!(
    test_cls_attribute_in_constructor,
    r#"
from typing import ClassVar
class A:
    def __new__(cls, x: int):
        cls.x = x
class B:
    def __init_subclass__(cls, x: int):
        cls.x = x
class C:
    x: ClassVar[int]
    def __new__(cls, x: int):
        cls.x = x
class D:
    x: ClassVar[int]
    def __init_subclass__(cls, x: int):
        cls.x = x
    "#,
);

testcase!(
    test_self_attribute_in_test_setup,
    r#"
class MyTestCase:
    def setUp(self):
        self.x = 5
    def run(self):
        assert self.x == 5
    "#,
);

testcase!(
    bug = "Attributes assigned in TestCase.setUpClass should be available on the class",
    test_class_attribute_in_setup_class,
    r#"
from unittest import TestCase

class Base(TestCase):
    shared: int

class Child(Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.shared = 1

    def test_shared(self) -> None:
        assert self.shared == 1

Child.shared
    "#,
);

testcase!(
    bug = "Example of how making methods read-write but not invariant is unsound",
    test_method_assign,
    r#"
from typing import Protocol
class X(Protocol):
    def foo(self) -> object:
        return 1
class Y:
    def foo(self) -> int:
        return 1
def func(x: X):
    x.foo = lambda: "hi"
y: Y = Y()
func(y)
y.foo()  # result is "hi"
    "#,
);

testcase!(
    test_attribute_union,
    r#"
class A:
    x: int
class B:
    x: str
def test(x: A | B):
    del x.x
    x.x = 1  # E: `Literal[1]` is not assignable to attribute `x` with type `str`
    "#,
);

testcase!(
    test_callable_boundmethod_subset,
    r#"
from typing import Callable

class C:
    def f(self, x: int, /) -> str:
        return ""
class C2:
    @classmethod
    def f(cls, x: int, /) -> str:
        return ""
class C3:
    @staticmethod
    def f(x: int, /) -> str:
        return ""
def foo(x: Callable[[int], str], c: C, c2: C2, c3: C3):
    C.f = x  # E: `(int) -> str` is not assignable to attribute `f` with type `(self: C, x: int, /) -> str`
    c.f = x
    C2.f = x
    c2.f = x
    C3.f = x
    c3.f = x
    "#,
);

testcase!(
    bug = "classmethod bound object w/o targs is default-instantiated, solves T to Any",
    test_bound_classmethod_explicit_targs,
    r#"
from typing import assert_type
class A[T]:
    x: T
    def __init__(self, x: T):
        self.x = x
    @classmethod
    def m(cls: 'type[A[T]]', x: T) -> 'A[T]':
        return cls(x)

assert_type(A[int].m(0), A[int])
assert_type(A.m(0), A[int]) # TODO # E: assert_type(A[Any], A[int]) failed

def test_typevar_bounds[T: A[int]](x: type[T]):
    assert_type(x.m(0), A[int])
    "#,
);

testcase!(
    test_use_of_class_body_scope_in_class_body_statement,
    r#"
class A:
    x: int = 5
    y: int = x
    "#,
);

testcase!(
    test_annotating_non_self_attributes,
    r#"
class A:
    x: int

class B:
    def __init__(self, a: A):
        a.x: int = 1  # E: Type cannot be declared in assignment to non-self attribute `a.x`

a: A = A()
a.x: int = 5  # E: Type cannot be declared in assignment to non-self attribute `a.x`
    "#,
);

testcase!(
    test_self_attribute_annotated_in_class_body,
    r#"
from typing import assert_type
class A:
    x: str
    def __init__(self, x: int):
        self.x = x  # E: `int` is not assignable to attribute `x` with type `str`
    "#,
);

testcase!(
    test_self_attribute_annotated_assignment,
    r#"
from typing import assert_type

class A:
    def __init__(self, x: str):
        self.x: int = x  # E: `str` is not assignable to attribute `x` with type `int`
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_generic_classvar,
    r#"
from typing import ClassVar
class A[T]:
    x: ClassVar[T]  # E: `ClassVar` arguments may not contain any type variables
    y: ClassVar[list[T]]  # E: `ClassVar` arguments may not contain any type variables
    "#,
);

testcase!(
    test_self_attribute_annotated_twice,
    r#"
from typing import assert_type, Literal, Final
class A:
    x: int
    y: str
    def __init__(self):
        self.x: Literal[1] = 1
        self.y: Final = "y"
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_final_attribute_assigned_in_init,
    r#"
from typing import assert_type, Final, Literal
class A:
    def __init__(self):
        self.x: Final = 0
def f(a: A):
    assert_type(a.x, Literal[0])
    "#,
);

testcase!(
    test_literal_attr_with_annotation,
    r#"
from typing import ClassVar, assert_type
class C:
    x0 = 0
    x1: ClassVar = 0
assert_type(C.x0, int)
assert_type(C.x1, int)
"#,
);

testcase!(
    test_final_annotated_override,
    r#"
from typing import Final
def f() -> int: ...
class Base:
    p: Final = f()
class Derived(Base):
    p = f()  # E: `p` is declared as final in parent class `Base`
"#,
);

testcase!(
    test_self_attribute_bare_annotation,
    r#"
from typing import assert_type
class A:
    def __init__(self, x: str):
        self.x: int
        self.x = x  # E: `str` is not assignable to attribute `x` with type `int`
def f(a: A):
    assert_type(a.x, int)
    "#,
);

testcase!(
    test_attribute_inference,
    r#"
class C:
    x: list[int | str]
def f(c: C):
    c.x = [5]
    "#,
);

testcase!(
    test_set_attribute_in_init_nested,
    r#"
from typing import assert_type
class C:
    def __init__(self):
        def f():
            self.x = 0
        f()
def f(c: C):
    assert_type(c.x, int)
    "#,
);

// TODO: Should we implement simple control-flow heuristics so `C.x` is recognized here?
testcase!(
    test_set_attribute_in_init_indirect,
    TestEnv::new().enable_implicitly_defined_attribute_error(),
    r#"
class C:
    def __init__(self):
        self.f()
    def f(self):
        self.x = 0  # E: Attribute `x` is implicitly defined by assignment in method `f`, which is not a constructor
def f(c: C) -> int:
    return c.x
    "#,
);

testcase!(
    test_missing_self_parameter,
    r#"
class C:
    def f():
        pass
C().f()  # E: Expected 0 positional arguments, got 1 (including implicit `self`)
    "#,
);

testcase!(
    test_generic_instance_method,
    r#"
class C:
    def f[T](self: T, x: T):
        pass
C().f(C())  # OK
C().f(0)    # E: Argument `Literal[0]` is not assignable to parameter `x` with type `C`
    "#,
);

// Make sure we treat `callable_attr` as a bare instance attribute, not a bound method.
testcase!(
    test_callable_instance_only_attribute,
    r#"
from typing import Callable, assert_type, Literal
class C:
    callable_attr: Callable[[int], int]
    def __init__(self):
       self.callable_attr = lambda x: x
c = C()
x = c.callable_attr(42)
assert_type(x, int)
    "#,
);

// To align with mypy & pyright, ClassVar[Callable] attributes should have method binding behavior
// See https://discuss.python.org/t/when-should-we-assume-callable-types-are-method-descriptors/92938
testcase!(
    test_callable_as_class_var,
    r#"
from typing import assert_type, Callable, ClassVar
def get_callback() -> Callable[[object, int], int]: ...
class C:
    f: ClassVar[Callable[[object, int], int]] = get_callback()
assert_type(C.f(None, 1), int)
assert_type(C().f(1), int)
"#,
);

// Mypy and Pyright treat `f` as not a method here; its actual behavior
// is ambiguous even if we assume the values are always functions or lambdas
// because the default value can be overridden by instance assignment.
//
// Our behavior is compatible, but the underlying implementation is not, we are
// behaving this way based on how we treat the Callable type rather than based
// on the absence of `ClassVar`.
//
// See https://discuss.python.org/t/when-should-we-assume-callable-types-are-method-descriptors/92938
testcase!(
    test_callable_with_ambiguous_binding,
    r#"
from typing import assert_type, Callable
def get_callback() -> Callable[[object, int], int]: ...
class C:
    f = get_callback()
assert_type(C.f(None, 1), int)
assert_type(C().f(None, 1), int)
# This is why the behavior is ambiguous - at runtime, the default `C.f` is a
# method but the instance-level shadow is not.
C().f = lambda _, x: x
"#,
);

testcase!(
    test_class_access_of_instance_only_attribute,
    r#"
from typing import assert_type, Any
class C:
    x: int
    def __init__(self, y: str):
        self.x = 0
        self.y = y
assert_type(C.x, int)
assert_type(C.y, Any)  # E: Instance-only attribute `y` of class `C` is not visible on the class
c = C("y")
assert_type(c.x, int)
assert_type(c.y, str)
"#,
);

testcase!(
    test_match_method_against_callable,
    r#"
from typing import Callable
class C:
    def f(self, x: int) -> None:
        pass
def f1(c: Callable[[int], None]):
    pass
def f2(c: Callable[[C, int], None]):
    pass
f1(C.f)  # E: Argument `(self: C, x: int) -> None` is not assignable to parameter `c` with type `(int) -> None`
f1(C().f)
f2(C.f)
f2(C().f)  # E: Argument `BoundMethod[C, (self: C, x: int) -> None]` is not assignable to parameter `c` with type `(C, int) -> None`
    "#,
);

testcase!(
    test_simple_inheritance,
    r#"
from typing import assert_type
class B:
    x: int

class HasBase(B):
    y: str

assert_type(HasBase().x, int)
"#,
);

testcase!(
    test_generic_multiple_inheritance,
    r#"
from typing import assert_type
class A[T]:
    x: T

class B[T]:
    y: T

class C[T](A[int], B[T]):
    z: bool

c: C[str] = C()
assert_type(c.x, int)
assert_type(c.y, str)
assert_type(c.z, bool)
"#,
);

testcase!(
    test_generic_chained_inheritance,
    r#"
from typing import assert_type
class A[T]:
    x: T

class B[T](A[list[T]]):
    y: T

class C[T](B[T]):
    z: bool

c: C[str] = C()
assert_type(c.x, list[str])
assert_type(c.y, str)
assert_type(c.z, bool)
"#,
);

testcase!(
    test_nested_class_attribute_with_inheritance,
    r#"
from typing import assert_type

class B:
    class Nested:
        x: int

class C(B):
    pass

N0: B.Nested = C.Nested()
N1: C.Nested = B.Nested()
assert_type(N1.x, int)
"#,
);

testcase!(
    test_class_generic_attribute_lookup,
    r#"
class C[T]:
    x: list[T] = []

C.x  # E: Generic attribute `x` of class `C` is not visible on the class
"#,
);

testcase!(
    test_var_attribute,
    r#"
from typing import assert_type
def f[T](x: T) -> T:
    return x
class C:
    def __init__(self):
        self.x = 42
assert_type(f(C()).x, int)
    "#,
);

testcase!(
    test_never_attr,
    r#"
from typing import Never, NoReturn, assert_type
def f() -> NoReturn: ...
def g():
    x = f().x
    assert_type(x, Never)
    "#,
);

testcase!(
    test_callable_attr,
    r#"
from typing import assert_type
from types import CodeType
def f():
    pass
def g():
    assert_type(f.__code__, CodeType)
    "#,
);

testcase!(
    test_boundmethod_attr,
    r#"
from typing import assert_type
class A:
    def f(self):
        pass
def g(a: A):
    assert_type(a.f.__self__, object)
    "#,
);

testcase!(
    test_ellipsis_attr,
    r#"
x = ...
x.x  # E: Object of class `EllipsisType` has no attribute `x`
    "#,
);

testcase!(
    test_forall_attr,
    r#"
from typing import assert_type
from types import CodeType
def f[T](x: T) -> T:
    return x
assert_type(f.__code__, CodeType)
    "#,
);

testcase!(
    test_metaclass_attr,
    r#"
from typing import assert_type

class A: ...
class B[T]: ...
assert_type(A.mro(), list[type])
assert_type(B[int].mro(), list[type])

class Meta(type):
    x: int
class C(metaclass=Meta):
    pass
assert_type(C.x, int)
    "#,
);

testcase!(
    test_metaclass_method_cls_typetype,
    r#"
from typing import assert_type

class Meta(type):
    def m[T](cls: type[T]) -> T: ...

class C(metaclass=Meta):
    pass

assert_type(C.m(), C)
"#,
);

fn env_with_stub() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "foo",
        "foo.pyi",
        r#"
class A:
    x: int = ...
    y: int
    "#,
    );
    t
}

testcase!(
    test_stub_initializes_attr,
    env_with_stub(),
    r#"
from typing import assert_type
from foo import A

assert_type(A.x, int)
assert_type(A.y, int)
    "#,
);

testcase!(
    test_object_getattribute,
    r#"
from typing import *
class A:
    def __getattribute__(self, name: str, /) -> int: ...
    def __setattr__(self, name: str, value: Any, /) -> None: ...
    def __delattr__(self, name: str, /) -> None: ...
class B:
    def __getattribute__(self, name: str, /) -> str: ...
a = A()
b = B()
assert_type(a.x, int)
assert_type(b.x, str)
a.x = 1
del a.x
b.x = 1  # E: Object of class `B` has no attribute `x`
del b.x  # E: Object of class `B` has no attribute `x`
    "#,
);

testcase!(
    test_object_getattr,
    r#"
from typing import assert_type

class Foo:
    def __getattr__(self, name: str) -> int: ...

def test(foo: Foo) -> None:
    assert_type(foo.x, int)
    assert_type(foo.y, int)
    foo.x = 1  # E: Object of class `Foo` has no attribute `x`
    del foo.y  # E: Object of class `Foo` has no attribute `y`
    "#,
);

testcase!(
    test_object_getattr_wrong_signature,
    r#"
from typing import assert_type

class Foo:
    def __getattr__(self, name: int) -> int: ...

def test(foo: Foo) -> None:
    assert_type(foo.x, int)  # E: Argument `Literal['x']` is not assignable to parameter `name`
    assert_type(foo.y, int)  # E: Argument `Literal['y']` is not assignable to parameter `name`
    foo.x = 1  # E: Object of class `Foo` has no attribute `x`
    del foo.y  # E: Object of class `Foo` has no attribute `y`
    "#,
);

testcase!(
    test_object_setattr,
    r#"
from typing import assert_type

class Foo:
    def __getattr__(self, name: str) -> int: ...
    def __setattr__(self, name: str, value: int) -> None: ...

def test(foo: Foo) -> None:
    foo.x = 1
    foo.x = ""  # E: Argument `Literal['']` is not assignable to parameter `value` with type `int`
    "#,
);

testcase!(
    test_object_delattr,
    r#"
from typing import assert_type

class Foo:
    def __getattr__(self, name: str) -> int: ...
    def __delattr__(self, name: str) -> None: ...

def test(foo: Foo) -> None:
    del foo.x
    "#,
);

testcase!(
    test_object_setattr_wrong_signature,
    r#"
from typing import assert_type

class Foo:
    def __getattr__(self, name: int) -> int: ...
    def __setattr__(self, name: int, value: int) -> None: ...  # E: `Foo.__setattr__` overrides parent class `object` in an inconsistent manner

def test(foo: Foo) -> None:
    foo.x = 1  # E: Argument `Literal['x']` is not assignable to parameter `name` with type `int`
    "#,
);

testcase!(
    test_argparse_namespace_setattr,
    r#"
from argparse import ArgumentParser, Namespace

ap: ArgumentParser = ArgumentParser()
ap.add_argument("-b", "--bool-flag", default=False, action='store_true')
ap.add_argument("-i", "--integer", default=1, type=int)
ap.add_argument("-s", "--string-arg", type=str, default="")
args: Namespace = ap.parse_args()
if not args.string_arg:
    args.string_arg = "string-goes-here"
    "#,
);

testcase!(
    test_module_getattr,
    TestEnv::one("foo", "def __getattr__(name: str) -> int: ..."),
    r#"
from typing import assert_type
import foo
assert_type(foo.x, int)
assert_type(foo.y, int)
foo.x = 1  # E: No attribute `x` in module `foo`
del foo.y  # E: No attribute `y` in module `foo`
    "#,
);

testcase!(
    test_any_subclass,
    r#"
from typing import Any, assert_type

class A(Any):
    x: int

class B(A):
    y: str

def test0(a: A, b: B, ta: type[A]) -> None:
    assert_type(a.x, int)
    assert_type(b.x, int)
    assert_type(b.y, str)
    assert_type(ta.mro(), list[type])

    assert_type(a.z, Any)
    assert_type(b.z, Any)
    assert_type(ta.z, Any)

class Test(B):
    def m(self) -> None:
        assert_type(super().z, Any)
    @classmethod
    def m2(cls) -> None:
        assert_type(super().z, Any)
    "#,
);

testcase!(
    test_field_using_method_scope_type_variable,
    r#"
from typing import assert_type, Any

class C:
    def __init__[R](self, field: R):
        self.field = field  # E: Attribute `field` cannot depend on type variable `R`, which is not in the scope of class `C`

c = C("test")
assert_type(c.field, Any)
"#,
);

// Note the difference between this and test_set_attribute_to_class_scope_type_variable.
// `R` in `__init__` here refers to a method-scoped type variable that shadows a class-scoped one.
testcase!(
    test_illegal_type_variable_with_name_shadowing,
    r#"
class C[R]:
    def __init__[R](self, field: R):  # E: Type parameter `R` shadows a type parameter of the same name from an enclosing scope
        self.field = field  # E: Attribute `field` cannot depend on type variable `R`, which is not in the scope of class `C`
"#,
);

// Note the difference between this and test_illegal_type_variable_with_name_shadowing.
// `R` in `__init__` here refers to the class-scoped `R``.
testcase!(
    test_set_attribute_to_class_scope_type_variable,
    r#"
from typing import Generic, TypeVar

R = TypeVar("R")

class C1(Generic[R]):
    def __init__(self, field: R):
        self.field = field

class C2[R]:
    def __init__(self, field: R):
        self.field = field
"#,
);

testcase!(
    test_attr_unknown,
    r#"
class Op:
    default: str
class Namespace:
    def __getattr__(self, op_name):
        if op_name == "__file__":
            return "test"
        return Op()
x = Namespace().some_op.default # E:  Object of class `str` has no attribute `default

"#,
);

testcase!(
    test_with,
    r#"
class C:
    def __init__(self) -> None:
        self.prev = False
    def __enter__(self) -> None:
        self.prev = False
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.prev = False
    def __new__(cls, orig_func=None):
        if orig_func is None:
            return super().__new__(cls)
def f():
    with C():  # E: `NoneType` has no attribute `__enter__`  # E: `NoneType` has no attribute `__exit__`
        pass
    "#,
);

testcase!(
    bug = "TODO(stroxler): We need to define the semantics of generic class nesting and avoid leaked type variables",
    test_class_nested_inside_generic_class,
    r#"
from typing import Any, assert_type, reveal_type
class Outer[T]:
    class Inner:
        x: T | None = None
assert_type(Outer[int].Inner, type[Outer.Inner])
assert_type(Outer.Inner, type[Outer.Inner])
reveal_type(Outer[int].Inner.x)  # E: revealed type: T | None
reveal_type(Outer.Inner.x)  # E: revealed type: T | None
reveal_type(Outer[int].Inner().x)  # E: revealed type: T | None
reveal_type(Outer.Inner().x)  # E: revealed type: T | None
   "#,
);

testcase!(
    test_attr_base,
    r#"
def f(x, key, dict):
    for param in x:
        if key in dict:
            pass
    "#,
);

testcase!(
    test_classvar_no_value,
    r#"
from typing import ClassVar, assert_type
class C:
    x: ClassVar[int]
assert_type(C.x, int)
    "#,
);

testcase!(
    test_set_attribute_on_typevar_annotated_self,
    r#"
from typing import Self, TypeVar
Self2 = TypeVar('Self2', bound='A')
class A:
    x: int
    def f1(self: Self, x: int, y: float) -> Self:
        self.x = x
        self.x = y  # E: `float` is not assignable to attribute `x` with type `int`
        return self
    def f2(self: Self2, x: int, y: float) -> Self2:
        self.x = x
        self.x = y  # E: `float` is not assignable to attribute `x` with type `int`
        return self
    def f3[Self3: A](self: Self3, x: int, y: float) -> Self3:
        self.x = x
        self.x = y  # E: `float` is not assignable to attribute `x` with type `int`
        return self
    "#,
);

testcase!(
    test_typevar_attr_default_error,
    r#"
from typing import Any, assert_type
class A[T1 = int, T2]:  # E:
    x: T2
def f(a: A):
    assert_type(a.x, Any)
    "#,
);

testcase!(
    test_union_base_class,
    r#"
from typing import Any, assert_type
class A:
    x: int
class B:
    x: str
try:
    C = A
except:
    C = B
class D(C): # E: Invalid base class: `A | B`
    pass
def f(d: D):
    assert_type(d.x, Any)
    "#,
);

testcase!(
    test_getattr_dispatch_for_metaclass,
    r#"
from typing import assert_type
class EMeta(type):
    def __getattr__(self, attr: str) -> int: ...
class E(int, metaclass=EMeta):
    pass
assert_type(E.EXAMPLE_VALUE, int)
    "#,
);

testcase!(
    test_getattr_selection_for_class_object_w_metaclass,
    r#"
from typing import assert_type
class EMeta(type):
    def __getattr__(self, attr: str) -> int: ...
class E(metaclass=EMeta):
    def __getattr__(self, attr: str) -> str: ...
assert_type(E.EXAMPLE_VALUE, int)
    "#,
);

testcase!(
    test_getattr_selection_for_class_object_no_metaclass,
    r#"
from typing import assert_type
class E:
    def __getattr__(self, attr: str) -> str: ...
E.EXAMPLE_VALUE # E: Class `E` has no class attribute `EXAMPLE_VALUE`
    "#,
);

testcase!(
    test_attribute_access_on_type_var,
    r#"
from typing import assert_type, Any
class Foo:
    def m(self) -> int:
        return 0
def f[T: Foo](y: T, z: type[T]) -> T:
    assert_type(y.m(), int)
    assert_type(z.m(y), int)
    assert_type(T.m(y), Any) # E: Object of class `TypeVar` has no attribute `m`
    return y
    "#,
);

testcase!(
    test_classmethod_on_instance_typevar_bound,
    r#"
from typing import Self

class A:
    @classmethod
    def bar(cls) -> Self: ...

def test[T: A](a: T) -> T:
    return a.bar()
    "#,
);

testcase!(
    test_attribute_access_on_quantified_bound_by_union,
    r#"
from typing import assert_type
class Foo:
    x: int
class Bar:
    x: str
def f[T: Foo | Bar](y: T, z: Foo | Bar) -> T:
    assert_type(z.x, int | str)
    assert_type(y.x, int | str)
    return y
    "#,
);

testcase!(
    test_attribute_access_on_type_none,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

ty(None).__bool__(None)
"#,
);

testcase!(
    test_attribute_access_on_type_literal,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

ty(0).bit_length(0)
"#,
);

testcase!(
    test_attribute_access_on_type_literalstring,
    r#"
from typing import LiteralString

# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

def test(x: LiteralString):
    ty(x).upper(x)
"#,
);

testcase!(
    test_attribute_access_on_type_callable,
    r#"
from typing import Callable

# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

def test_callable(x: Callable[[], None]):
    ty(x).__call__(x)
"#,
);

testcase!(
    test_attribute_access_on_type_function,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

def foo(): ...

ty(foo).__call__(foo)
"#,
);

testcase!(
    test_attribute_access_on_type_boundmethod,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

class X:
    def m(self): ...

ty(X().m).__call__(X().m)
"#,
);

testcase!(
    test_attribute_access_on_type_overload,
    r#"
from typing import overload

# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

@overload
def bar(x: int) -> int: ...
@overload
def bar(x: str) -> str: ...
def bar(x: int | str) -> int | str: ...

ty(bar).__call__(bar)
"#,
);

testcase!(
    test_attribute_access_on_type_union,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

class A:
    x = 0
class B:
    x = "foo"

def test_union(x: A  | B):
    ty(x).x
"#,
);

testcase!(
    bug = "type[ClassDef(..)] and type[ClassType(..)] should be type (or the direct metaclass?)",
    test_attribute_access_on_type_class,
    r#"
# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

class C:
    @staticmethod
    def m(x: int): ...

class D[T]:
    @classmethod
    def m(cls, x: T): ...

ty(C).m(0) # E: Expr::attr_infer_for_type attribute base undefined
ty(D[int]).m(0) # E: Expr::attr_infer_for_type attribute base undefined
"#,
);

testcase!(
    bug = "type[TypedDict()] should be type",
    test_attribute_access_on_type_typeddict,
    r#"
from typing import TypedDict

# handy hack to get a type[X] for any X
def ty[T](x: T) -> type[T]: ...

class TD(TypedDict):
    x: int

ty(TD(x = 0))(x = 0)
ty(TD).mro() # E: Expr::attr_infer_for_type attribute base undefined
"#,
);

testcase!(
    test_type_magic_dunder_compare,
    r#"
def test(x: type[int], y: type[int]) -> None:
    # These are OK because `type` inherits `__eq__` and `__ne__` from `object`.
    x == y
    x != y

    # These are always OK
    x is y
    x is not y

    # These are not OK because the corresponding dunder methods are not defined on `type`
    x < y       # E: `<` is not supported between `type[int]` and `type[int]`
    x <= y      # E: `<=` is not supported between `type[int]` and `type[int]`
    x > y       # E: `>` is not supported between `type[int]` and `type[int]`
    x >= y      # E: `>=` is not supported between `type[int]` and `type[int]`
    x in y      # E: `in` is not supported between `type[int]` and `type[int]`
    x not in y  # E: `not in` is not supported between `type[int]` and `type[int]`
    "#,
);

testcase!(
    test_access_method_using_class_param_on_class,
    r#"
from typing import assert_type, reveal_type, Any
class A[T]:
    def f(self) -> T: ...
reveal_type(A.f) # E: revealed type: [T](self: A[T]) -> T
assert_type(A.f(A[int]()), int)
    "#,
);

testcase!(
    test_access_generic_method_using_class_param_on_class,
    r#"
from typing import assert_type, reveal_type, Any
class A[T]:
    def f[S](self, x: S) -> tuple[S, T]: ...
reveal_type(A.f) # E: revealed type: [T, S](self: A[T], x: S) -> tuple[S, T]
assert_type(A.f(A[int](), ""), tuple[str, int])
    "#,
);

testcase!(
    test_access_overloaded_method_using_class_param_on_class,
    r#"
from typing import assert_type, reveal_type, overload, Any
class A[T]:
    @overload
    def f(self) -> T: ...
    @overload
    def f(self, x: T | None) -> T: ...
    def f(self, x=None) -> Any: ...
reveal_type(A.f) # E: revealed type: Overload[[T](self: A[T]) -> T, [T](self: A[T], x: T | None) -> T]
assert_type(A.f(A[int]()), int)
    "#,
);

testcase!(
    test_access_overloaded_staticmethod_using_class_param_on_class,
    r#"
from typing import assert_type, reveal_type, overload, Any
class A[T]:
    @overload
    @staticmethod
    def f(x: None = ...) -> None: ...
    @overload
    @staticmethod
    def f(x: T) -> T: ...
    @staticmethod
    def f(x = None) -> Any: ...
reveal_type(A.f) # E: revealed type: Overload[(x: None = ...) -> None, [T](x: T) -> T]
assert_type(A.f(), None)
assert_type(A.f(0), int)
    "#,
);

testcase!(
    test_invalid_augmented_assign_in_init,
    r#"
class C:
    def __init__(self):
        self.x += 5  # E: Object of class `C` has no attribute `x`
    "#,
);

testcase!(
    test_attributes_when_raw_class_field_type_contains_var,
    r#"
from typing import assert_type, Any
# This test is making sure we don't leak a `Var` into a ClassField, which can lead to nondeterminism.
class A:
    x = []
    y = []
assert_type(A().x, list[Any])
A().x = [42]
A().y = [42]
assert_type(A().y, list[Any])
    "#,
);

testcase!(
    test_read_only_frozen_dataclass,
    r#"
import dataclasses

@dataclasses.dataclass(frozen=True)
class FrozenData:
    x: int
    y: str

def f(d: FrozenData):
    d.x = 42  # E: Cannot set field `x`
    d.y = "new"  # E: Cannot set field `y`
    "#,
);

testcase!(
    test_read_only_namedtuple,
    r#"
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int

def f(p: Point):
    p.x = 10  # E: Cannot set field `x`
    p.y = 20  # E: Cannot set field `y`
    "#,
);

testcase!(
    test_read_only_annotation_typeddict,
    r#"
from typing_extensions import TypedDict, ReadOnly

class Config(TypedDict):
    name: ReadOnly[str]
    value: int

def f(c: Config):
    c["value"] = 42  # OK
    c["name"] = "new"  # E: Key `name` in TypedDict `Config` is read-only
    "#,
);

testcase!(
    test_nested_class_mutability,
    r#"
class Backend:
    class Options:
        pass
class Options2(Backend.Options):
    pass
Backend.Options = Options2  # E: A class object initialized in the class body is considered read-only
    "#,
);

testcase!(
    test_nested_class_inheritance,
    r#"
class Backend:
    class Options:
        pass
class ProcessGroupGloo(Backend):
    class Options(Backend.Options):
        pass
    "#,
);

testcase!(
    test_nested_class_inheritance_via_assignment,
    r#"
class Backend:
    class Options:
        pass
class Options2(Backend.Options):
    pass
class ProcessGroupGloo(Backend):
    Options = Options2
    "#,
);

testcase!(
    test_read_only_class_var,
    r#"
from typing import ClassVar, Final
class C:
    x: ClassVar[Final[int]] = 42
C.x = 43  # E: This field is marked as Final
    "#,
);

testcase!(
    test_final_qualifier_with_inherited_type,
    r#"
from typing import Final
class Parent:
    x: float = 1
class Child(Parent):
    x: Final = 2  # E: `Child.x` is read-only, but `Parent.x` is read-write
child = Child()
child.x = 3.0  # E: Cannot set field `x`
    "#,
);

testcase!(
    test_inherited_annotation_with_tuple_unpacking,
    r#"
from typing import assert_type
class Parent:
    x: float
    y: float
class Child(Parent):
    x, y = 3, 4
child = Child()
assert_type(child.x, float)
assert_type(child.y, float)
    "#,
);

testcase!(
    test_attr_cast,
    r#"
from typing import Self, cast, Any, assert_type

class C:
    outputs: list[Any]
    def f(self, other):
        other = cast(Self, other)
        assert_type(other, Self)
        assert_type(other.outputs, list[Any])
        len(self.outputs) == len(other.outputs)
    "#,
);

testcase!(
    test_attr_tuple,
    r#"
from typing import Any, Tuple

def g(ann) -> None:
    if ann is Tuple: ...
    ann.__module__
    "#,
);

testcase!(
    test_tuple_attribute_example,
    r#"
def f(obj, g, field_type, my_type,):
    assert issubclass(obj, tuple) and hasattr(obj, "_fields")
    for f in obj._fields:
        if isinstance(field_type, my_type) and g is not None:
            if g is None:
                raise ValueError(
                    f"{obj.__name__}."
                )
    "#,
);

testcase!(
    test_set_attr_in_child_class,
    r#"
from typing import assert_type

class A:
    def __init__(self):
        self.x = 0

class B(A):
    def f(self):
        self.x = ""  # E: `Literal['']` is not assignable to attribute `x` with type `int`

class C(A):
    def f(self):
        self.x: str = ""  # E: Class member `C.x` overrides parent class `A` in an inconsistent manner
    "#,
);

testcase!(
    test_method_sets_inherited_generic_field,
    r#"
# Regression test for a bug tracked in https://github.com/facebook/pyrefly/issues/774
from typing import assert_type, Any
class A[T]:
    x: T
class B(A[int]):
    def __init__(self, x: int):
        # The test is primarily verifying that we handle this implicit definition
        # correctly in the class field logic, when this is actually an inherited field.
        self.x = x
assert_type(B(42).x, int)
    "#,
);

testcase!(
    test_private_attr_assignment_in_constructor,
    r#"
from typing import assert_type

class Config:
    pass

class DerivedConfig(Config):
    def foo(self) -> None:
        print("hello")

class B:
    def __init__(self, config: Config) -> None:
        self.__config = config

class C(B):
    def __init__(self, config: DerivedConfig) -> None:
        self.__config = config

    def bar(self) -> None:
        assert_type(self.__config, DerivedConfig)
        self.__config.foo()
    "#,
);

testcase!(
    test_crtp_example, // CRTP = Curiously recurring template pattern
    r#"
from typing import Any, assert_type
class Node[T: "Node[Any]"]:
    children : tuple[T, ...]
class Expr(Node["Expr"]):
    ...
class Singleton(Expr):
    def __init__(self, v: Expr):
        self.children = (v,)
assert_type(Singleton(Expr()).children, tuple[Expr, ...])
    "#,
);

testcase!(
    test_mro_method,
    r#"
().mro()  # E: no attribute `mro`
tuple.mro()
type.mro()  # E: Missing argument `self`
    "#,
);

// How special forms are represented in typing.py is an implementation detail, but in practice,
// some of the representations are stable across Python versions. In particular, user code
// sometimes relies on some special forms being classes and Type behaving like builtins.type.
testcase!(
    test_special_forms,
    r#"
from typing import Callable, Generic, Protocol, Tuple, Type
def f1(cls):
    if cls is Callable:
        return cls.mro()
def f2(cls):
    if cls is Generic:
        return cls.mro()
def f3(cls):
    if cls is Protocol:
        return cls.mro()
def f4(cls):
    if cls is Tuple:
        return cls.mro()
def f5(cls, x: type):
    if cls is Type:
        return cls.mro(x)
    "#,
);

testcase!(
    test_get_type_new,
    r#"
from typing import cast, reveal_type
def get_type_t[T]() -> type[T]:
    return cast(type[T], 0)
def foo[T](x: type[T]):
    # mypy reveals the same thing we do (the type of `type.__new__`), while pyright reveals `Unknown`.
    reveal_type(get_type_t().__new__)  # E: Overload[[Self@type](cls: type[Self@type], o: object, /) -> type, [Self](cls: type[Self], name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwds: Any) -> Self]
    "#,
);

testcase!(
    test_any_value_lookup,
    r#"
from typing import Any
Any.foo.bar # E: Class `Any` has no class attribute `foo`
    "#,
);

// T, P, and Ts are values of type TypeVar, ParamSpec, and TypeVarTuple respectively.
// They should behave like values when we try to access attributes on them.
testcase!(
    test_typevar_value_lookups,
    r#"
from typing import Callable, TypeVar, ParamSpec, TypeVarTuple

def ty[T](x: T) -> type[T]: ...

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")

T.__name__
P.__name__
P.args.__origin__
P.kwargs.__origin__
Ts.__name__

ty(T).__name__
ty(P).__name__
ty(P.args).__origin__
ty(P.args).__origin__
ty(Ts).__name__

def f(g: Callable[P, T], ts: tuple[*Ts], *args: P.args, **kwargs: P.kwargs):
    args.count(1)
    kwargs.keys()

    ty(args).count(args, 1)
    ty(kwargs).keys(kwargs)

    T.__name__
    P.__name__
    P.args.__origin__
    P.kwargs.__origin__
    Ts.__name__

    ty(T).__name__
    ty(P).__name__
    ty(P.args).__origin__
    ty(P.args).__origin__
    ty(Ts).__name__
"#,
);

testcase!(
    test_type_never,
    r#"
from typing import Never, assert_type, reveal_type
def f() -> type[Never]: ...
reveal_type(f().mro) # E: BoundMethod[type, (self: type) -> list[type]]
assert_type(f().wut, Never)
    "#,
);

testcase!(
    bug = "We should note when a classmethod creates an implicit attribute that captures a type parameter",
    test_implicit_class_attribute_captures_method_tparam,
    r#"
from typing import reveal_type
class A:
    @classmethod
    def f[T](cls, x: T):
        cls.x = x
reveal_type(A.x)  # E: revealed type: T
    "#,
);

testcase!(
    test_lazy_class_attribute_init,
    r#"
from typing import assert_type
class C:
    @classmethod
    def m(cls):
        if hasattr(cls, "foo"):
            return cls.foo
        retval = "foo"
        cls.foo = retval
        return retval
assert_type(C.foo, str)
    "#,
);

// See https://github.com/facebook/pyrefly/issues/1448 for what this tests
// and discussion of approaches to handling `@functools.wraps` with return
// type inference.
testcase!(
    test_inferred_returns_from_functools_wraps,
    r#"
from typing import assert_type, Any
from functools import wraps
def decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return wrapper
class C:
    @decorator
    def f(self) -> int: ...
assert_type(C().f(), Any)
    "#,
);

testcase!(
    test_missing_attribute_suggests_similar_name,
    r#"
class Foo:
    value = 1

def f(obj: Foo) -> None:
    obj.vaule  # E: Object of class `Foo` has no attribute `vaule`\n  Did you mean `value`?
"#,
);

testcase!(
    test_missing_attribute_suggests_builtin_str_method,
    r#"
"".lowerr  # E: Object of class `str` has no attribute `lowerr`\n  Did you mean `lower`?
"#,
);

testcase!(
    test_missing_attribute_suggests_inherited,
    r#"
class Base:
    field = 1

class Child(Base):
    pass

def f(x: Child) -> None:
    x.filed  # E: Object of class `Child` has no attribute `filed`\n  Did you mean `field`?
"#,
);

testcase!(
    test_missing_attribute_suggests_typed_dict_field,
    r#"
from typing import TypedDict
class TD(TypedDict):
    foo: int

def f(x: TD) -> int:
    return x.fo  # E: Object of class `TD` has no attribute `fo`\n  Did you mean `foo`?
"#,
);

testcase!(
    test_missing_attribute_suggests_enum_member,
    r#"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def f(x: Color) -> Color:
    return x.BLU  # E: Object of class `Color` has no attribute `BLU`\n  Did you mean `BLUE`?
"#,
);

testcase!(
    test_class_toplevel_inherited_attr_name,
    r#"
# at the class top level, inherited attribute names should be considered in scope
from typing import assert_type

class Foo:
    assert_type(__qualname__, str)
    assert_type(__module__, str)
    attr: int

class Bar(Foo):
    assert_type(attr, int)
    "#,
);

testcase!(
    test_set_attr_to_none,
    r#"
from typing import Any, assert_type
class A:
    def __init__(self):
        self.x = None
        self.y: None = None
    def set_x(self, x: int):
        self.x = x
    def set_y(self, y: int):
        self.y = y  # E: `int` is not assignable to attribute `y` with type `None`
def f(a: A):
    assert_type(a.x, Any | None)
    assert_type(a.y, None)
    "#,
);
