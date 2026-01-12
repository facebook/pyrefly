/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_primitive_subtyping,
    r#"
class A: pass
class B(A): pass
class C(B): pass
class D: pass

b: A = B()
c: A = C()
oops: A = D()  # E: `D` is not assignable to `A`
"#,
);

testcase!(
    test_object_is_top,
    r#"
class A: pass

a: object = A()
s: object = ""
"#,
);

testcase!(
    test_simple_generic_subtyping,
    r#"
class A[T]: pass
class B[T](A[T]): pass
class C(B[int]): pass
class D[T]: pass

b: A[int] = B[int]()
c: A[int] = C()
oops1: A[int] = D[int]()  # E: `D[int]` is not assignable to `A[int]`
# Although T is bivariant in A, we follow mypy and pyright's lead in treating it as invariant.
oops2: A[int] = A[str]()  # E: `A[str]` is not assignable to `A[int]`
"#,
);

testcase!(
    test_simple_class_object_subtyping,
    r#"
class A: pass
class B(A): pass

a: type[A] = B
b: type[B] = A  # E: `type[A]` is not assignable to `type[B]`
"#,
);

testcase!(
    test_generic_class_object_subtyping,
    r#"
class A[T]: pass
class B[T](A[T]): pass

a0: type[A] = B
b0: type[B] = A  # E: `type[A]` is not assignable to `type[B[Unknown]]`

a1: type[A[int]] = B
b1: type[B[int]] = A  # E: `type[A]` is not assignable to `type[B[int]]`

a2: type[A] = B[int]
b2: type[B] = A[int]  # E: `type[A[int]]` is not assignable to `type[B[Unknown]]`
"#,
);

testcase!(
    test_literal_string_subtyping,
    r#"
from typing import Literal, LiteralString

def l0() -> Literal["foo"]: ...
def l1() -> Literal["foo"] | Literal["bar"]: ...
def l2() -> LiteralString: ...
def l3() -> LiteralString | Literal["foo"]: ...
def l4() -> str: ...

test0: LiteralString = l0()
test1: LiteralString = l1()
test2: str = l0()
test3: str = l1()
test4: str = l2()
test5: Literal["foo"] = l2()  # E: `LiteralString` is not assignable to `Literal['foo']`
test6: str = l3()
test7: LiteralString = l4()  # E: `str` is not assignable to `LiteralString`

test10: object = l0()
test11: object = l3()
"#,
);

testcase!(
    test_type_guard_subtyping,
    r#"
from typing import Callable, TypeGuard, Any

def t0(x: Any) -> TypeGuard[bool]: ...
def t1(x: Any) -> TypeGuard[int]: ...
def t2(x: Any) -> TypeGuard[bool] | TypeGuard[int]: ...
def t3(x: Any) -> bool: ...

test0: Callable[[Any], bool] = t0
test1: Callable[[Any], TypeGuard[int]] = t0
test2: Callable[[Any], bool] = t1
test3: Callable[[Any], TypeGuard[bool]] = t1  # E: `(x: Any) -> TypeGuard[int]` is not assignable to `(Any) -> TypeGuard[bool]`
test4: Callable[[Any], bool] = t2
test5: Callable[[Any], TypeGuard[int]] = t2
test6: Callable[[Any], TypeGuard[bool]] = t2  # E: `(x: Any) -> TypeGuard[bool] | TypeGuard[int]` is not assignable to `(Any) -> TypeGuard[bool]
test7: Callable[[Any], TypeGuard[bool]] = t3  # E: `(x: Any) -> bool` is not assignable to `(Any) -> TypeGuard[bool]`

test8: Callable[[Any], object] = t0
test9: Callable[[Any], object] = t2
"#,
);

testcase!(
    test_type_is_subtyping,
    r#"
from typing import Callable, TypeIs, Any

def t0(x: Any) -> TypeIs[bool]: ...
def t1(x: Any) -> TypeIs[int]: ...
def t2(x: Any) -> TypeIs[bool] | TypeIs[int]: ...
def t3(x: Any) -> bool: ...

test0: Callable[[Any], bool] = t0
test1: Callable[[Any], TypeIs[int]] = t0  # E: `(x: Any) -> TypeIs[bool]` is not assignable to `(Any) -> TypeIs[int]`
test2: Callable[[Any], bool] = t1
test4: Callable[[Any], bool] = t2
test5: Callable[[Any], TypeIs[int]] = t3  # E: `(x: Any) -> bool` is not assignable to `(Any) -> TypeIs[int]`

test6: Callable[[Any], object] = t0
test7: Callable[[Any], object] = t2
"#,
);

// See https://typing.python.org/en/latest/spec/generics.html#user-defined-generic-types.
// When no type argument is specified when inheriting from a generic class, we use the default or Any.
// In particular, the bound is not used.
testcase!(
    test_subclass_generic_no_targ,
    r#"
from typing import Any, assert_type
class A[T: float]:
    x: T
class B[T: float = int]:
    x: T
class AChild(A):
    pass
class BChild(B):
    pass
def f(a: AChild, b: BChild):
    assert_type(a.x, Any)
    assert_type(b.x, int)
    "#,
);

testcase!(
    test_extends_type,
    r#"
class A(type): pass
def test(a: A):
    x: type[int] = a
"#,
);

testcase!(
    test_subclass_generic_missing_targs,
    r#"
from typing import Any, assert_type
class A[T1, T2, T3 = int]:
    x1: T1
    x2: T2
    x3: T3
class B(A[str]):  # E: Expected 3 type arguments for `A`, got 1
    pass
def f(b: B):
    assert_type(b.x1, str)
    assert_type(b.x2, Any)
    assert_type(b.x3, int)
    "#,
);

testcase!(
    test_pyo3_enum_pattern,
    r#"
# This pattern is not valid runtime Python as written because you cannot
# use a class before its body has finished evaluating, the name isn't
# defined.
#
# But it's possible for non-declarative Python code to behave like this,
# which means it is useful if we recognize it in stubs. Perhaps more
# importantly, this is how PyO3, the most popular rust-to-python conversion
# tool, always represents Rust enums.
#
# // (In the rust code from which these stubs come):
# #[pyclass]
# #[derive(Serialize, Deserialize, Debug, Clone)]
# pub enum PyO3Enum {
#   Variant0 {},
#   Variant1 {},
# }
class PyO3Enum:
    class Variant0(PyO3Enum): ...
    class Variant1(PyO3Enum): ...

x: PyO3Enum = PyO3Enum.Variant0()
    "#,
);

// See https://github.com/facebook/pyrefly/issues/622 for an example
// of when it matters to have reasonable analysis of `type(...)` in
// base classes; sometimes the runtime semantics require this, particularly
// dealing with C extensions.

testcase!(
    bug = "We probably need to be able to handle `type(...)` as a base class better than we do.",
    test_type_function_in_base_class_list_v0,
    r#"
class A:
    pass
a = A()
class B(type(a)):  # E: Invalid expression form for base class: `type(a)`
    pass
    "#,
);

testcase!(
    bug = "We probably need to be able to handle `type(...)` as a base class better than we do.",
    test_type_function_in_base_class_list_v1,
    r#"
class A:
    pass
class B(type(A)):  # E: Invalid expression form for base class: `type(A)`
    pass
    "#,
);

testcase!(
    bug = "We probably need to be able to handle `type(...)` as a base class better than we do.",
    test_type_function_in_base_class_list_v2,
    r#"
from typing import assert_type, ClassVar, Any
class M(type):
    x: ClassVar[int] = 42
class A(metaclass=M):
    pass
class B(type(A)):  # E: Invalid expression form for base class: `type(A)`
    pass
assert_type(B.x, Any)
    "#,
);

testcase!(
    test_multiple_inheritance_incompatible_field,
    r#"
class Foo:
    p: int
class Bar:
    p: str

class Both(Foo, Bar): # E: Field `p` has inconsistent types inherited from multiple base classes
    ...
"#,
);

testcase!(
    test_nested_multiple_inheritance_incompatible_field_without_override,
    r#"
class A:
    x: int
class B:
    x: str
class C(A, B): # E: Field `x` has inconsistent types inherited from multiple base classes
    pass
class D:
    x: int

# We do not report the error for E, since it has already been reported on C
class E(C, D):
    pass
"#,
);

testcase!(
    test_nested_multiple_inheritance_incompatible_field_with_override,
    r#"
class A:
    x: int
class B:
    x: str
class C(A, B):
    x: int # E: Class member `C.x` overrides parent class `B` in an inconsistent manner
class D:
    x: int

# We do not report the error on E, since we already reported an error on C
class E(C, D):
    pass
"#,
);

testcase!(
    test_multiple_inheritance_incompatible_methods,
    r#"
class Foo:
    def foo(self) -> int: ...
class Bar:
    def foo(self) -> str: ...

class Both(Foo, Bar): # E: Field `foo` has inconsistent types inherited from multiple base classes
    ...
"#,
);

testcase!(
    test_multiple_inheritance_compatible_generic_methods,
    r#"
class Foo[T1]:
    def foo(self) -> T1: ...
class Bar[T2]:
    def foo(self) -> T2: ...

class Both[T](Foo[T], Bar[T]): # Should have no error here
    ...
"#,
);

testcase!(
    test_multiple_inheritance_special_methods,
    r#"
class Foo:
    def __init__(self, x: int) -> None: ...
class Bar:
    def __init__(self, x: str) -> None: ...

class Both(Foo, Bar): # No error here, because __init__ is special
    ...
"#,
);

testcase!(
    test_multiple_inheritance_read_write,
    r#"
class Foo:
    x: int
    @property
    def y(self) -> int:
        return 1
class Bar:
    x: float
    @property
    def y(self) -> float:
        return 1

# For read-write fields, the inherited type from each parent should be assignable to the intersection
class Both(Foo, Bar):  # E: Field `x` is declared `float`
    ...
"#,
);

fn env_conditional_import() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
class Foo1: pass
class Foo2: pass
"#,
    )
}

testcase!(
    test_conditional_import_base_class,
    env_conditional_import(),
    r#"
if int("1"):
    from foo import Foo1 as Foo
else:
    from foo import Foo2 as Foo

class Document(Foo): pass  # E: Invalid base class: `Foo1 | Foo2`

from abc import ABC, abstractmethod
class CustomModel[T: Document](ABC):
    @abstractmethod
    async def to_db(self) -> T:
        pass
"#,
);

testcase!(
    test_multiple_inheritance_property,
    r#"
from typing import overload

class A:
    @property
    def x(self, /) -> int: ...
    @x.setter
    def x(self, x: int, /) -> None: ...

class B:
    @property
    def x(self, /) -> int: ...

class C(A, B): ...

class D:
    @property
    def x(self, /) -> str: ...
    @x.setter
    def x(self, x: str, /) -> None: ...

class E(D, B): ...  # E: Field `x` has inconsistent types inherited from multiple base classes
"#,
);
