/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
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
    test_generic_bounds,
    r#"
class A: ...
class B(A): ...
class C(B): ...

def test[T: B](x: T) -> None:
    a: A = x  # OK
    b: B = x  # OK
    c: C = x  # E: `T` is not assignable to `C`

test(A())  # E: `A` is not assignable to upper bound `B` of type variable `T`
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
class Bar(Foo[A]):  # E: `A` is not assignable to upper bound `B` of type variable `T`
    pass
 "#,
);

testcase!(
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

test(A())  # E: `A` is not assignable to upper bound `B | C` of type variable `T`
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
class Bar(Foo[A]):  # E: `A` is not assignable to upper bound `B | C` of type variable `T`
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
    test_enum_bound_subscript,
    r#"
from enum import Enum
from typing import assert_type

def takes_enum[T: Enum](what: type[T]) -> T:
    return what["abc"]

class SomeEnum(Enum):
    abc = 1

assert_type(takes_enum(SomeEnum), SomeEnum)
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
assert_type(C.attr, int)  # E: assert_type(Any, int) failed  # E: Generic attribute `attr` of class `C` is not visible on the class
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

testcase!(
    test_typevar_single_constraint_is_error,
    r#"
from typing import TypeVar

T = TypeVar("T", int)  # E: Expected at least 2 constraints in TypeVar `T`, got 1
    "#,
);

testcase!(
    test_typevar_literal_bound,
    r#"
from typing import Literal, LiteralString, assert_type
def f[T: Literal["foo"]](x: T) -> T: ...
def g[T: LiteralString](x: T) -> T: ...

assert_type(f("foo"), Literal["foo"])
assert_type(f("bar"), str) # E: `str` is not assignable to upper bound `Literal['foo']`

assert_type(g("foo"), Literal["foo"])
assert_type(g("bar"), Literal["bar"])
    "#,
);

testcase!(
    bug = "This should succeed with no errors",
    test_add_with_constraints,
    r#"
def add[T: (int, str)](x: T, y: T) -> T:
    return x + y # E: `+` is not supported # E: `+` is not supported # E: `int | Unknown` is not assignable to declared return type `T`
    "#,
);

testcase!(
    bug = "Unexpected error in add3",
    test_add_with_upper_bound,
    r#"
# This is not allowed because it's legal to pass something like `add1(0, "1")` to this function.
def add1[T: int | str](x: T, y: T):
    return x + y # E: `+` is not supported # E: `+` is not supported

# This is ok.
def add2[T: int](x: T, y: T) -> int:
    return x + y

# This is also ok.
def add3[T: int | float](x: T, y: T) -> int | float:
    return x + y # E: `+` is not supported
    "#,
);

testcase!(
    bug = "Spurious '`+` is not supported' error",
    test_add_with_upper_bound_and_bad_return_type,
    r#"
# mypy and pyright both reject both of these, so we do, too.
def add1[T: int](x: T, y: T) -> T:
    return x + y # E: Returned type `int` is not assignable to declared return type `T`
def add2[T: int | float](x: T, y: T) -> T:
    return x + y # E: `+` is not supported # E: Returned type `float | int` is not assignable to declared return type `T`
    "#,
);

testcase!(
    bug = "This should succeed with no errors. Pyrefly pins the types too early due to https://github.com/facebook/pyrefly/issues/105",
    test_multiple_args_upper_bound,
    r#"
from typing import assert_type

def f[T: int | str](x: T, y: T): ...
f(0, "1") # E: `Literal['1']` is not assignable to parameter `y` with type `int`

class A[T]:
    def __init__(self, x: T, y: T): ...
# Note: pyright says the type is A[int | str]; mypy says A[object].
# Either is okay, but A[int] is definitely wrong and we shouldn't emit the not assignable error.
assert_type(A(0, "1"), A[int | str]) # E: assert_type(A[int], A[int | str]) # E: `Literal['1']` is not assignable to parameter `y` with type `int`
    "#,
);

testcase!(
    test_use_default_for_unsolved_typevar_in_function_with_infer_with_first_use,
    TestEnv::new_with_infer_with_first_use(true),
    r#"
from typing import assert_type
def f[T = int]() -> T: ...
assert_type(f(), int)
    "#,
);

testcase!(
    test_use_default_for_unsolved_typevar_in_function_no_infer_with_first_use,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import assert_type
def f[T = int]() -> T: ...
assert_type(f(), int)
    "#,
);

testcase!(
    test_arg_against_typevar_bound,
    r#"
from typing import Callable, Iterable
def reduce[_S](function: Callable[[_S, _S], _S], iterable: Iterable[_S]) -> _S: ...
def f[_T: str](arg1: _T, arg2: _T) -> _T: ...
reduce(f, [1])  # E: `int` is not assignable to upper bound `str` of type variable `_T`
reduce(f, ["ok"])
    "#,
);

testcase!(
    test_catch_typevar_default_violation,
    r#"
def f[T = int]() -> list[T]: ...
x = f()
x.append("oops")  # E: `Literal['oops']` is not assignable to parameter `object` with type `int`
    "#,
);

testcase!(
    test_typevar_with_default_or_none,
    r#"
from typing import assert_type
def f[T1, T2 = str](x: T1, y: T2 | None = None) -> T1 | T2: ...
assert_type(f(1), int | str)
    "#,
);

testcase!(
    test_unsolved_typevar_with_constraints,
    r#"
from typing import TypeVar
T = TypeVar("T", int, str)
def f(t: T | None) -> None:...
f(None)
    "#,
);

testcase!(
    test_circular_bound,
    r#"
from typing import Any, assert_type

class Node[P]: ...

class A[P: A](Node[P]):
    def __init__(self, x: P | None = None):
        self.x = x
    def f(self) -> P: ...

a1 = A()
assert_type(a1, A[Any])
assert_type(a1.f(), Any)

a2 = A(a1)
assert_type(a2, A[A[Any]])
assert_type(a2.f(), A[Any])
    "#,
);

testcase!(
    test_circular_bound_with_imported_base,
    TestEnv::one("node", "class Node[P]: ..."),
    r#"
from node import Node
from typing import Any, assert_type

class A[P: A](Node[P]):
    def __init__(self, x: P | None = None):
        self.x = x
    def f(self) -> P: ...

a1 = A()
assert_type(a1, A[Any])
assert_type(a1.f(), Any)

a2 = A(a1)
assert_type(a2, A[A[Any]])
assert_type(a2.f(), A[Any])
    "#,
);

testcase!(
    test_typevar_default_referencing_typevar,
    r#"
from typing import TypeAlias, Any, assert_type
from typing_extensions import Generic, TypeVar

class NBitBase:
    pass

class _32Bit(NBitBase):
    pass

_NBit1 = TypeVar("_NBit1", bound=NBitBase, default=Any)
_NBit2 = TypeVar("_NBit2", bound=NBitBase, default=_NBit1)

class complexfloating(Generic[_NBit1, _NBit2]): ...
complex64: TypeAlias = complexfloating[_32Bit]

def f(z: complex64):
    assert_type(z, complexfloating[_32Bit, _32Bit])
"#,
);

// PEP 696 validation tests for TypeVar defaults when default is another TypeVar
testcase!(
    test_typevar_default_bound_validation,
    r#"
from typing import TypeVar

class A: ...
class B(A): ...
class C(B): ...

# When default is a TypeVar, T1's bound must be a subtype of T2's bound

# Valid: default TypeVar's bound (B) is subtype of outer bound (A)
T1 = TypeVar("T1", bound=B)
T2 = TypeVar("T2", bound=A, default=T1)

# Valid: same bound is OK
T3 = TypeVar("T3", bound=A)
T4 = TypeVar("T4", bound=A, default=T3)

# Invalid: default TypeVar's bound (A) is NOT a subtype of outer bound (B)
T5 = TypeVar("T5", bound=A)
T6 = TypeVar("T6", bound=B, default=T5)  # E: Expected default `TypeVar[T5]` of `T6` to be assignable to the upper bound of `B`

# Valid: default TypeVar has narrower bound (C) which is subtype of (A)
T7 = TypeVar("T7", bound=C)
T8 = TypeVar("T8", bound=A, default=T7)
"#,
);

testcase!(
    test_typevar_default_constraints_validation,
    r#"
from typing import TypeVar

class A: ...
class B: ...
class C: ...
class D: ...

# When default is a TypeVar, the outer constraints must be a superset of default's constraints

# Valid: outer constraints (A, B, C) is superset of default constraints (A, B)
T1 = TypeVar("T1", A, B)
T2 = TypeVar("T2", A, B, C, default=T1)

# Valid: same constraints is OK
T3 = TypeVar("T3", A, B)
T4 = TypeVar("T4", A, B, default=T3)

# Invalid: outer constraints (A, B) is NOT a superset of default constraints (A, B, C)
T5 = TypeVar("T5", A, B, C)
T6 = TypeVar("T6", A, B, default=T5)  # E: Expected default `TypeVar[T5]` of `T6` to be one of the following constraints: `A`, `B`

# Invalid: outer constraints (A,) does not include all of default constraints (A, B)
T7 = TypeVar("T7", A, B)
T8 = TypeVar("T8", A, C, default=T7)  # E: Expected default `TypeVar[T7]` of `T8` to be one of the following constraints: `A`, `C`
"#,
);

testcase!(
    test_typevar_default_mixed_restriction_validation,
    r#"
from typing import TypeVar

class A: ...
class B: ...
class C: ...

# A bounded TypeVar cannot be a valid default for a constrained TypeVar
T1 = TypeVar("T1", bound=A)
T2 = TypeVar("T2", A, B, default=T1)  # E: Expected default `TypeVar[T1]` of `T2` to be one of the following constraints: `A`, `B`

# An unrestricted TypeVar cannot be a valid default for a constrained TypeVar
T3 = TypeVar("T3")
T4 = TypeVar("T4", A, B, default=T3)  # E: Expected default `TypeVar[T3]` of `T4` to be one of the following constraints: `A`, `B`

# An unrestricted TypeVar can be valid for a bounded TypeVar (unrestricted means bound=object)
T5 = TypeVar("T5")
T6 = TypeVar("T6", bound=object, default=T5)  # OK - unrestricted bound is object
"#,
);

testcase!(
    test_typevar_default_typevar_pep695_syntax,
    r#"
from typing import assert_type

class A: ...
class B(A): ...
class C(B): ...

# Test with PEP 695 syntax (new generic syntax)

# Valid: default TypeVar bound is subtype of outer bound
class Container1[T1: B, T2: A = T1]: ...
x1: Container1[C] = Container1()
assert_type(x1, Container1[C, C])

# Invalid: default TypeVar bound is NOT subtype of outer bound
class Container2[T1: A, T2: B = T1]: ...  # E: Expected default `T1` of `T2` to be assignable to the upper bound of `B`
"#,
);

testcase!(
    test_typevar_default_typevar_constraints_pep695_syntax,
    r#"
class A: ...
class B: ...
class C: ...

# Test constrained TypeVar defaults with PEP 695 syntax

# Valid: outer constraints are superset of default constraints
class Container1[T1: (A, B), T2: (A, B, C) = T1]: ...

# Invalid: outer constraints are NOT superset of default constraints
class Container2[T1: (A, B, C), T2: (A, B) = T1]: ...  # E: Expected default `T1` of `T2` to be one of the following constraints: `A`, `B`
"#,
);
