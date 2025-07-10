/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

// T is bivariant in A since it's not used nor in a covariant nor contravariant position.
testcase!(
    test_simple_generic_subtyping,
    r#"
class A[T]: pass
class B[T](A[T]): pass
class C(B[int]): pass
class D[T]: pass

b: A[int] = B[int]()
c: A[int] = C()
oops: A[int] = D[int]()  # E: `D[int]` is not assignable to `A[int]`
ok: A[int] = A[str]()
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

// See https://github.com/facebook/pyrefly/issues/622
testcase!(
    bug = "We probably need to be able to handle `type(...)` as a base class.",
    test_bad_gradual_is_subset_eq,
    r#"
from typing import Generic
class CustomMetaclass(type(Generic)):  # E: Expected a type form, got instance of `type`
    pass
    "#,
);
