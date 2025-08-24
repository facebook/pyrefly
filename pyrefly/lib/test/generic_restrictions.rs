/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_quantified_subtyping_no_constraint,
    r#"
def test[T](x: T) -> None:
    y: int = x  # E: `T` is not assignable to `int`
    z: object = x  # OK
 "#,
);

testcase!(
    test_type_var_tuple_default,
    r#"
from typing import TypeVarTuple, Unpack, assert_type

Ts1 = TypeVarTuple("Ts1", default=Unpack[tuple[int, int]])
Ts2 = TypeVarTuple("Ts2", default=int)  # E: Default for `TypeVarTuple` must be an unpacked tuple form or another `TypeVarTuple`, got `int`

def test[*Ts = Unpack[tuple[int, int]]](x: tuple[*Ts]) -> tuple[*Ts]:
    return x
def test2[*Ts = int](x: tuple[*Ts]) -> tuple[*Ts]:  # E: Default for `TypeVarTuple` must be an unpacked tuple form or another `TypeVarTuple`, got `int`
    return x

class C[*Ts = Unpack[tuple[int, int]]]:
    def foo(self) -> tuple[*Ts]: ...
assert_type(C().foo(), tuple[int, int])
 "#,
);

testcase!(
    test_param_spec_default,
    r#"
from typing import ParamSpec, Callable

P1 = ParamSpec("P1", default=...)
P2 = ParamSpec("P2", default=[str, int])
P3 = ParamSpec("P3", default=int)  # E: Default for `ParamSpec` must be a parameter list, `...`, or another `ParamSpec`, got `int`

def test[**P = ...](x: Callable[P, None]) -> Callable[P, None]:
    return x
def test2[**P = [str, int]](x: Callable[P, None]) -> Callable[P, None]:
    return x
 "#,
);

testcase!(
    test_var_subtype_deadlock,
    r#"
from typing import Iterator

def iter_iter[T](x: Iterator[T]) -> Iterator[T]:
    return iter(x)

iter_iter(iter([1, 2, 3]))
 "#,
);

testcase!(
    bug = "Instantiation is not validated against bound, see https://github.com/facebook/pyrefly/issues/111",
    test_generic_bounds,
    r#"
class A: ...
class B(A): ...
class C(B): ...

def test[T: B](x: T) -> None:
    a: A = x  # OK
    b: B = x  # OK
    c: C = x  # E: `T` is not assignable to `C`

test(A())  # Not OK
test(B())
test(C())
 "#,
);

testcase!(
    test_base_class_bound,
    r#"
class A: pass
class B: pass

class Foo[T: B]:
    pass

class Bar(Foo[B]):  # OK
    pass
class Bar(Foo[A]):  # E: Type `A` is not assignable to upper bound `B` of type variable `T`
    pass
 "#,
);

testcase!(
    bug = "Instantiation is not validated against constraints, see https://github.com/facebook/pyrefly/issues/111",
    test_generic_constraints,
    r#"
class A: ...
class B(A): ...
class C(A): ...
class D(C): ...

def test[T: (B, C)](x: T) -> None:
    a: A = x  # OK
    b: B = x  # E: `T` is not assignable to `B`
    c: C = x  # E: `T` is not assignable to `C`
    d: B | C = x  # OK

test(A())  # Not OK
test(B())
test(C())
test(D())
 "#,
);

testcase!(
    test_base_class_constraint,
    r#"
class A: pass
class B: pass
class C: pass

class Foo[T: (B, C)]:
    pass

class Bar(Foo[B]):  # OK
    pass
class Bar(Foo[A]):  # E: Type `A` is not assignable to upper bound `B | C` of type variable `T`
    pass
 "#,
);

testcase!(
    test_generic_constraint_with_default,
    r#"
from typing import TypeVar
class A: ...
class B(A): ...
class C(A): ...
class D(C): ...

def test1[T: (B, C) = A](x: T) -> None:  # E: Expected default `A` of `T` to be one of the following constraints: `B`, `C`
    pass
def test2[T: (B, C) = B](x: T) -> None:
    pass
def test3[T: (B, C) = C](x: T) -> None:
    pass
def test4[T: (B, C) = D](x: T) -> None:  # E: Expected default `D` of `T` to be one of the following constraints: `B`, `C`
    pass

T1 = TypeVar("T1", B, C, default=A)  # E: Expected default `A` of `T1` to be one of the following constraints: `B`, `C`
T2 = TypeVar("T2", B, C, default=B)
T3 = TypeVar("T3", B, C, default=C)
T4 = TypeVar("T4", B, C, default=D)  # E: Expected default `D` of `T4` to be one of the following constraints: `B`, `C`
 "#,
);

testcase!(
    test_generic_bound_with_default,
    r#"
from typing import TypeVar
class A: ...
class B(A): ...
class C(A): ...
class D(C): ...

def test1[T: C = A](x: T) -> None:  # E: Expected default `A` of `T` to be assignable to the upper bound of `C`
    pass
def test2[T: C = B](x: T) -> None:  # E: Expected default `B` of `T` to be assignable to the upper bound of `C`
    pass
def test3[T: C = C](x: T) -> None:
    pass
def test4[T: C = D](x: T) -> None:
    pass

T1 = TypeVar("T1", bound=C, default=A)  # E: Expected default `A` of `T1` to be assignable to the upper bound of `C`
T2 = TypeVar("T2", bound=C, default=B)  # E: Expected default `B` of `T2` to be assignable to the upper bound of `C`
T3 = TypeVar("T3", bound=C, default=C)
T4 = TypeVar("T4", bound=C, default=D)
 "#,
);

testcase!(
    test_bounded_callable,
    r#"
from typing import Callable, TypeVar, assert_type
T = TypeVar('T', bound=Callable[[int], int])
def func(a: T, b: int) -> T:
    assert_type(a(b), int)
    return a
T2 = TypeVar('T2', Callable[[int], int], Callable[[int], bool])
def func2(a: T2, b: int) -> T2:
    assert_type(a(b), int | bool)
    return a
 "#,
);

testcase!(
    test_bounded_callable_protocol,
    r#"
from typing import Protocol, TypeVar, Self, assert_type
class A(Protocol):
    def __call__(self) -> Self: ...
class B(Protocol):
    def __call__(self) -> Self: ...
T = TypeVar('T', bound=A | B)
def func(a: T) -> T:
    return a()
T2 = TypeVar('T2', A, B)
def func2(a: T2) -> T2:
    return a()
 "#,
);

testcase!(
    test_bounded_typevar_attribute_access,
    r#"
from typing import TypeVar, assert_type
class C:
    x: int
T = TypeVar('T', bound=C)
def func(c: T) -> C:
    assert_type(c.x, int)
    return c
 "#,
);

testcase!(
    test_instantiate_default_typevar,
    r#"
from typing import assert_type, reveal_type, Callable, Self
class C[T = int]:
    def meth(self, /) -> Self:
        return self
    attr: T
reveal_type(C.meth)  # E: [T](self: C[T], /) -> C[T]
assert_type(C.attr, int)  # E: assert_type(Any, int) failed  # E: Instance-only attribute `attr` of class `C` is not visible on the class
 "#,
);

testcase!(
    test_union_bound_attr_get,
    r#"
from typing import assert_type
class A:
    x: int
class B:
    x: str
def f[T: A | B](x: T) -> T:
    assert_type(x.x, int | str)
    return x
    "#,
);

testcase!(
    test_constraints_attr_get,
    r#"
from typing import assert_type
class A:
    x: int
class B:
    x: str
def f[T: (A, B)](x: T) -> T:
    assert_type(x.x, int | str)
    return x
    "#,
);

testcase!(
    test_unrestricted_attr_get,
    r#"
from typing import assert_type
def f[T](x: T) -> T:
    assert_type(x.__doc__, str | None)
    x.nonsense # E: `object` has no attribute `nonsense`
    return x
    "#,
);

testcase!(
    test_pass_along_bounded_typevar,
    r#"
from typing import TypeVar
T = TypeVar('T', bound='A')
class A:
    def f(self: T) -> T:
        return self
    def g(self: T) -> T:
        return self.f()
    "#,
);

testcase!(
    test_preserve_generic_self,
    r#"
class A:
    def m[S: A](self: S) -> S:
        return self
def g[T: A](a: T) -> T:
    return a.m()
    "#,
);

testcase!(
    test_pass_along_constrained_typevar,
    r#"
from typing import Self, TypeVar

class B():
    def f(self) -> Self:
        return self 
class C(B):
    pass
class D(B):
    pass

T = TypeVar( "T", C, D)
def g(b: T) -> T:
    return b.f()
    "#,
);

testcase!(
    test_constrained_typevar_attr_access,
    r#"
class A:
    x: int
class B:
    x: int
class Foo[T: (A, B)]:
    y: T
    def foo(self) -> None:
        self.y.__class__
    "#,
);

testcase!(
    test_bounded_type_of_type_var_access,
    r#"
from typing import Self, TypeVar, Protocol

class CustomCreation(Protocol):
    @classmethod
    def get_instance(cls) -> Self: ...

T = TypeVar("T", bound=CustomCreation)

def foo(val: type[T]) -> T:
    return val.get_instance()
    "#,
);

testcase!(
    test_constrained_typevar_protocol_subtype,
    r#"
from typing import Protocol
class P(Protocol):
    x: int
class A:
    x: int
class B:
    x: int
class Foo[T: (A, B)]:
    y: T
    def foo(self) -> None:
        p: P = self.y
    "#,
);

testcase!(
    test_constrained_typevar_mutate_attr,
    r#"
class A:
    x: int
class B:
    x: int
class Foo[T: (A, B)]:
    y: T
    def foo(self) -> None:
        self.y.x = 1
        self.y.x = ""  # E: `Literal['']` is not assignable to attribute `x` with type `int`
        del self.y.x
    "#,
);

testcase!(
    test_union_bounded_typevar_with_property_get,
    r#"
# https://github.com/facebook/pyrefly/issues/869
from collections import defaultdict
from typing import assert_type

class A:

    @property
    def attr(self) -> str:
        return "A"


class B:

    @property
    def attr(self) -> str:
        return "B"


def foo[T: A | B](items: list[T]) -> None:
    results: defaultdict[str, list[T]] = defaultdict(list)
    for item in items:
        assert_type(item.attr, str)
        results[item.attr].append(item)
    "#,
);

testcase!(
    test_union_bounded_typevar_property_return_self,
    r#"
from typing import Self
class A:
    @property
    def attr(self) -> Self: ...

class B:
    @property
    def attr(self) -> Self: ...

def foo[T: A | B](x: T) -> T:
    return x.attr
    "#,
);

testcase!(
    test_constrained_typevar_property_return_self,
    r#"
from typing import Self
class A:
    @property
    def attr(self) -> Self: ...

class B:
    @property
    def attr(self) -> Self: ...

def foo[T: (A, B)](x: T) -> T:
    return x.attr
    "#,
);

testcase!(
    test_union_bounded_typevar_instance_method_return_self,
    r#"
from typing import Self
class A:

    def method(self) -> Self: ...

class B:

    def method(self) -> Self: ...

def foo[T: A | B](x: T) -> T:
    return x.method()
    "#,
);

testcase!(
    test_constrained_typevar_instance_method_return_self,
    r#"
from typing import Self
class A:

    def method(self) -> Self: ...

class B:

    def method(self) -> Self: ...

def foo[T: (A, B)](x: T) -> T:
    return x.method()
    "#,
);
