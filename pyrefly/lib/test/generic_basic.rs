/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    bug =
        "We should use the bounds/constraints of the type var to determine the callable input type",
    test_tyvar_constructor,
    r#"
def test[T](cls: type[T]) -> T:
    cls(1)  # Not OK, we should assume object constructor here
    return cls()
class A:
    def __init__(self, x: int) -> None: pass
def test2[T: A](cls: type[T]) -> T:
    a1: A = cls()  # Not OK
    a2: A = cls(1)
    return cls()
"#,
);

testcase!(
    test_tyvar_mix,
    r#"
from typing import TypeVar, assert_type
U = TypeVar("U")
def foo[T](
      x: U  # E: Type parameter U is not included in the type parameter list
    ) -> U:
    return x

assert_type(foo(1), int)
"#,
);

// This test exercises an edge case where naively using type analysis on base classes
// can cause problems in the interaction of tparams validation and recursion.
testcase!(
    test_generic_with_reference_to_self_in_base,
    r#"
from typing import Generic, TypeVar, Any, assert_type

T = TypeVar("T")

class C(list[C[T]]):
    t: T

def f(c: C[int]):
    assert_type(c.t, int)
    assert_type(c[0], C[int])
    "#,
);

testcase!(
    test_redundant_generic_base,
    r#"
from typing import Generic
class C[T](Generic[T]):  # E: Redundant
    pass
    "#,
);

testcase!(
    test_class_type_params_can_reference_class,
    r#"
class C[T: C](set[object]):
    pass
    "#,
);

testcase!(
    test_type_argument_error_default,
    r#"
from typing import Any, assert_type
class C[T1, *Ts, T2]: pass
C_Alias = C[int]  # E: Expected 3 type arguments for `C`, got 1
assert_type(C[int], type[C[int, *tuple[Any, ...], Any]])  # E: Expected 3 type arguments for `C`, got 1

AnyClassMethod = classmethod[Any]  # E: Expected 3 type arguments for `classmethod`, got 1
assert_type(classmethod[Any], type[classmethod[Any, ..., Any]])  # E: Expected 3 type arguments for `classmethod`, got 1

# No error if it's a TypeVarTuple w/ nothing after, because a TypeVarTuple can be empty
class C2[T, *Ts]: pass
C2_Alias = C2[int]
assert_type(C2[int], type[C2[int, *tuple[Any, ...]]])
"#,
);

testcase!(
    bug = "T is pinned prematurely due to https://github.com/facebook/pyrefly/issues/105",
    test_generics,
    r#"
from typing import Literal
class C[T]: ...
def append[T](x: C[T], y: T):
    pass
v: C[int] = C()
append(v, "test")  # E: `Literal['test']` is not assignable to parameter `y` with type `int`
"#,
);
testcase!(
    test_generic_default,
    r#"
from typing import assert_type
class C[T1, T2 = int]:
    pass
def f9(c1: C[int, str], c2: C[str]):
    assert_type(c1, C[int, str])
    assert_type(c2, C[str, int])
    "#,
);

testcase!(
    test_generic_type,
    r#"
from typing import assert_type, Any
class A: ...
class B: ...
class C[T]: ...
class D[T = A]: ...
def f[E](e: type[E]) -> E: ...
assert_type(f(A), A)
assert_type(f(B), B)
assert_type(f(C), C[Any])
assert_type(f(D), D)
"#,
);

testcase!(
    test_untype_with_missing_targs_annotation,
    TestEnv::new().enable_implicit_any_error(),
    r#"
class C[T]: pass

x: C        # E: Cannot determine the type parameter `T` for generic class `C`
y: C | int  # E: Cannot determine the type parameter `T` for generic class `C`
z: list[C]  # E: Cannot determine the type parameter `T` for generic class `C`
    "#,
);

testcase!(
    test_untype_with_missing_targs_base_class,
    TestEnv::new().enable_implicit_any_error(),
    r#"
class C[T]: pass
class D(C): pass  # E: Cannot determine the type parameter `T` for generic class `C`
x: D
    "#,
);

testcase!(
    test_typevar_default_contains_nested_typevar,
    r#"
from typing import assert_type, TypeVar
class A[T1 = float, T2 = list[T1]]: pass
def f(a1: A[int], a2: A):
    assert_type(a1, A[int, list[int]])
    assert_type(a2, A[float, list[float]])
    "#,
);

// Test that we get the most precise type arguments we can even in the presence of errors.
testcase!(
    test_typevar_default_contains_typevar_error,
    r#"
from typing import Any, assert_type
class A[T1, T2 = int, T3, T4 = T1]:  # E: `T3` without a default cannot follow type parameter `T2` with a default
    pass
def f(a: A[str]):  # E: Expected 4 type arguments for `A`, got 1
    assert_type(a, A[str, int, Any, str])
    "#,
);

// This isn't allowed because it's ambiguous how many type arguments the TypeVarTuple consumes.
testcase!(
    test_typevar_with_default_after_typevartuple,
    r#"
from typing import assert_type, Any, reveal_type
class A[*Ts, T = int]:  # E: TypeVar `T` with a default cannot follow TypeVarTuple `Ts`
    pass
class B[*Ts, T1, T2 = T1]:  # E: TypeVar `T2` with a default cannot follow TypeVarTuple `Ts`
    pass
assert_type(B[int](), B[*tuple[()], int, int])
assert_type(B[int, str](), B[*tuple[()], int, str])
assert_type(B[int, str, float, bool, bytes](), B[int, str, float, bool, bytes])
# It doesn't matter too much how we fill in the type arguments when they aren't
# pinned by construction, as long as it's plausible.
reveal_type(B()) # E: revealed type: B[@_, @_, @_]
b: B[tuple[tuple[Any, ...], Any, Any]] = B()  # Here's one valid way to pin them
    "#,
);

testcase!(
    test_paramspec_with_default_after_typevartuple,
    r#"
from typing import Any, assert_type
class A[*Ts, **P1, **P2 = P1]:
    pass
class B[*Ts, T, **P = [int, str]]:
    pass
assert_type(A[[int, str]](), A[*tuple[()], [int, str], [int, str]])
assert_type(A[bool, [int, str]](),  A[bool, [int, str], [int, str]])
assert_type(A[bool, bytes, [int, str]](), A[bool, bytes, [int, str], [int, str]])
assert_type(B[int, str, float](), B[int, str, float, [int, str]])
    "#,
);

testcase!(
    bug = "should raise an error on bad_curry",
    test_functools_partial_pattern,
    r#"
from typing import Any, Callable, Concatenate, Generic, ParamSpec, TypeVar, TypeVarTuple, overload

_P1 = ParamSpec("_P1")
_P2 = ParamSpec("_P2")
_T = TypeVar("_T")
_R_co = TypeVar("_R_co", covariant=True)
_Ts = TypeVarTuple("_Ts")

class partial(Generic[_P1, _P2, _T, _R_co, *_Ts]):
    @overload
    def __new__(cls, __func: Callable[_P1, _R_co]) -> partial[_P1, _P1, Any, _R_co]: ...
    @overload
    def __new__(cls, __func: Callable[Concatenate[*_Ts, _P2], _R_co], *args: *_Ts) -> partial[Concatenate[*_Ts, _P2], _P2, Any, _R_co, *_Ts]: ...
    @overload
    def __new__(cls, __func: Callable[_P1, _R_co], *args: *_Ts, **kwargs: _T) -> partial[_P1, ..., _T, _R_co, *_Ts]: ...
    def __new__(cls, __func, *args, **kwargs):
        return super().__new__(cls)
    def __call__(self, *args: _P2.args, **kwargs: _P2.kwargs) -> _R_co: ...

def many_params(a: int, b: str, c: int, d: str) -> tuple[int, str]:
    return a + c, b + d

o1: tuple[int, str] = many_params(1, 'a', 2, 'b')

curry = partial(many_params, 17, 'foo')
o2a = curry(42, 'bar')

bad_curry = partial(many_params, 1, 'a', 2, 'b', 3, 'c', 4, 'd')
o2b = bad_curry(7, 11)
    "#,
);

testcase!(
    test_typevartuple_default_is_typevartuple,
    r#"
from typing import TypeVarTuple, Unpack
Ps = TypeVarTuple('Ps')
Qs = TypeVarTuple('Qs', default=Unpack[Ps])
# This error is expected. What we're testing is that the unpacked TypeVarTuple default is accepted
# without any additional error.
class A[*Ps, *Qs = *Ps]: # E: may not have more than one TypeVarTuple
    pass
    "#,
);

testcase!(
    test_specialize_error,
    r#"
from nowhere import BrokenGeneric, BrokenTypeVar # E: Could not find import of `nowhere`

class MyClass(BrokenGeneric[BrokenTypeVar]):
    pass

# We don't know how many type arguments to expect, since we have errors in the base type, so accept any number
def f(x: MyClass[int]):
    pass

# We should still report other errors in type arguments
def g(
    x: MyClass["NotAClass"],  # E: Could not find name `NotAClass`
    y: MyClass[0],  # E: Expected a type form, got instance of `Literal[0]`
):
    pass
"#,
);

testcase!(
    test_type_var_subtype_with_constraints,
    r#"
from typing import TypeVar, Generic

_b = TypeVar("_b", bool, int)
class F(Generic[_b]):
    def f(self, b: _b = True) -> _b: ...
    "#,
);

testcase!(
    bug = "Update should know about string arguments",
    test_dict_update,
    r#"
# From https://github.com/facebook/pyrefly/issues/245
from typing import assert_type, Any

def f():
    x = {}
    x.update(a = 1)
    assert_type(x, dict[str, int])

def g():
    x: dict[int, int] = {}
    x.update(a = 1) # E: No matching overload
"#,
);

testcase!(
    test_use_of_bad_generic,
    r#"
from typing import Generic
class C(Generic[oops]):  # E:
    pass
def f(c: C[int]):
    pass
    "#,
);

// Test various things that we should allow `type` to be specialized with
testcase!(
    test_type_argument_for_type,
    r#"
from typing import Any, TypeVar

class A: ...
class B: ...

a: type[A]
b: type[B]
c: type[A | B]
d: type[Any]

T1 = TypeVar('T1')
def f(x: type[T1]) -> T1:
    return x()

def g[T2](x: type[T2]) -> T2:
    return x()
    "#,
);

testcase!(
    test_generic_return_union,
    r#"
from typing import *

def hello[T](x: T) -> None | T:
    return x
"#,
);

testcase!(
    test_quantified_accumulation,
    TestEnv::one("foo", "import typing\nT = typing.TypeVar('T')"),
    r#"
from typing import reveal_type, TypeVar
from foo import T as TT

T = TypeVar("T")

def cond() -> bool:
    return True

def union[A, B](a: A, b: B) -> A | B:
    return a if cond() else b

def f(x: T, y: TT):
    a = union(x, y)
    a = union(a, a)
    a = union(a, a)
    reveal_type([a]) # E: revealed type: list[T]
"#,
);

testcase!(
    test_forall_matches_forall,
    r#"
from typing import Callable, Protocol
class Identity(Protocol):
    def __call__[T](self, x: T, /) -> T:
        return x
def f[T]() -> Callable[[T], T]:
    return lambda x: x
x: Identity = f()
    "#,
);

testcase!(
    test_too_new_syntax,
    TestEnv::new_with_version(PythonVersion::new(3, 8, 0)),
    r#"
class A[T]:  # E: Cannot use type parameter lists on Python 3.8 (syntax was added in Python 3.12)
    x: T
    "#,
);

testcase!(
    test_shadowing_scoped_type_vars,
    r#"
from typing import TypeVar, Generic
class C0[T]:
    def foo[T](self, x: T) -> T:  # E: Type parameter `T` shadows a type parameter of the same name from an enclosing scope
        return x
T = TypeVar("T")
class C1(Generic[T]):
    def foo[T](self, x: T) -> T:  # E: Type parameter `T` shadows a type parameter of the same name from an enclosing scope
        return x
    "#,
);

testcase!(
    test_typevar_or_none,
    r#"
from typing import assert_type
def f[T1, T2](x: T1, y: T2 | None = None) -> T1 | T2: ...
assert_type(f(1), int)
assert_type(f(1, "2"), int | str)
assert_type(f(1, None), int)
    "#,
);

testcase!(
    test_typevar_solved_in_one_path,
    r#"
from typing import assert_type
def f[T1, T2](x: T1, y: T2 | None, z: T2) -> T1 | T2: ...
assert_type(f(1, None, ""), int | str)
    "#,
);

testcase!(
    test_return_only_unsolved_typevars,
    r#"
from typing import Any, assert_type

def f1[T](x: T | None = None) -> T: ...
assert_type(f1(), Any)

def f2[T1, T2](x: T1 | None = None, y: T2 | None = None) -> T1 | T2: ...
assert_type(f2(), Any)
    "#,
);

testcase!(
    test_unsolved_typevar_multiple_occurrences,
    r#"
from typing import Any, assert_type
def f[T](x: T | None = None) -> tuple[T, T | int]: ...
assert_type(f(), tuple[Any, int])
    "#,
);

testcase!(
    test_pass_tuple_literal_through_identity_function,
    r#"
def f[T](x: T) -> T:
    return x
def g() -> int:
    return f((1, "hello world"))[0]
    "#,
);

testcase!(
    test_type_attr,
    r#"
from typing import assert_type
def f[T](
    config_type: type[T],
) -> T:
    assert_type(config_type.__name__, str)
    return config_type()
    "#,
);

testcase!(
    test_nested_typevar,
    r#"
from typing import assert_type
def f[T](x: list[T] | list[None], y: list[T]) -> T:
    return y[0]
assert_type(f([None], [0]), int)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1675
testcase!(
    test_generic_should_equal_itself,
    r#"
from typing import cast, Iterator, Any
def condition() -> bool: ...
def f[T](iterator: Iterator[T]) -> T:
    res = cast(T, None)
    while condition():
        for i in iterator:
            res = i
    return res
    "#,
);

testcase!(
    test_bounded_type_var_subscriptable,
    r#"
from collections.abc import Sequence

def test[S: Sequence[int]](sequence: S) -> int:
    return sequence[0]

def test2[S](not_a_sequence: S) -> int:
    return not_a_sequence[0]  # E: `S` is not subscriptable
    "#,
);

testcase!(
    test_generator_iterable,
    r#"
from typing import Any

type TypeForm[T] = type[T] | Any

def _to_list[T](
    value: Any,
    kind: type[list[T]] = list,
) -> list[T]:
    return kind(to_type(val, Any) for val in value)

def to_type[T](value: Any, kind: TypeForm[T]) -> T: ...
    "#,
);
