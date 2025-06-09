/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_tyvar_function,
    r#"
from typing import TypeVar, assert_type

T = TypeVar("T")

def foo(x: T) -> T:
    y: T = x
    return y

assert_type(foo(1), int)
"#,
);

testcase!(
    test_tyvar_alias,
    r#"
from typing import assert_type
import typing

T = typing.TypeVar("T")

def foo(x: T) -> T:
    return x

assert_type(foo(1), int)
"#,
);

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
    test_tyvar_quoted,
    r#"
from typing import assert_type
import typing

T = typing.TypeVar("T")

def foo(x: "T") -> "T":
    return x

assert_type(foo(1), int)
"#,
);

testcase!(
    test_tyvar_mix,
    r#"
from typing import TypeVar, assert_type
U = TypeVar("U")
def foo[T](
      x: U  # E: Type parameter U is not included in the type parameter list
    ) -> U:  # E: Type parameter U is not included in the type parameter list
    return x

assert_type(foo(1), int)
"#,
);

testcase!(
    test_legacy_generic_syntax,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")

class C(Generic[T]):
    x: T

c: C[int] = C()
assert_type(c.x, int)
    "#,
);

testcase!(
    test_legacy_generic_syntax_inheritance,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

class C(Generic[T]):
    x: T

class D(Generic[S], C[list[S]]):
    pass

d: D[int] = D()
assert_type(d.x, list[int])
    "#,
);

testcase!(
    test_legacy_generic_syntax_inherit_twice,
    r#"
from typing import Generic, TypeVar
_T = TypeVar('_T')
class A(Generic[_T]):
    pass
class B(A[_T]):
    pass
class C(B[int]):
    pass
    "#,
);

testcase!(
    test_legacy_generic_syntax_multiple_implicit_tparams,
    r#"
from typing import Generic, TypeVar, assert_type
_T = TypeVar('_T')
_U = TypeVar('_U')
class A(Generic[_T]):
    a: _T
class B(Generic[_T]):
    b: _T
class C(Generic[_T]):
    c: _T
class D(A[_T], C[_U], B[_T]):
    pass
x: D[int, str] = D()
assert_type(x.a, int)
assert_type(x.b, int)
assert_type(x.c, str)
    "#,
);

testcase!(
    test_legacy_generic_syntax_filtered_tparams,
    r#"
from typing import Generic, TypeVar
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
class A(Generic[_T1, _T2]):
    pass
class B(A[_T1, int]):
    pass
class C(B[str]):
    pass
    "#,
);

testcase!(
    bug = "The TODO here is because we implemented but have temporarily disabled a check for the use of a generic class without type arguments as a type annotation; this check needs to be configurable and we don't have the plumbing yet.",
    test_legacy_generic_syntax_implicit_targs,
    r#"
from typing import Any, Generic, TypeVar, assert_type
T = TypeVar('T')
class A(Generic[T]):
    x: T
def f(a: A):  # TODO: The generic class `A` is missing type arguments.
    assert_type(a.x, Any)
    "#,
);

testcase!(
    test_tvar_missing_name,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar()  # E: Missing `name` argument
P = ParamSpec()  # E: Missing `name` argument
Ts = TypeVarTuple()  # E: Missing `name` argument
    "#,
);

testcase!(
    test_tvar_wrong_name,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar("Z")  # E: TypeVar must be assigned to a variable named `Z`
P = ParamSpec("Z")  # E: ParamSpec must be assigned to a variable named `Z`
Ts = TypeVarTuple("Z")  # E: TypeVarTuple must be assigned to a variable named `Z
    "#,
);

testcase!(
    test_tvar_wrong_name_expr,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar(17)  # E: Expected first argument of TypeVar to be a string literal
P = ParamSpec(17)  # E: Expected first argument of ParamSpec to be a string literal
Ts = TypeVarTuple(17)  # E: Expected first argument of TypeVarTuple to be a string literal
    "#,
);

testcase!(
    test_tvar_wrong_name_bind,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
x = "test"
T = TypeVar(x)  # E: Expected first argument of TypeVar to be a string literal
P = ParamSpec(x)  # E: Expected first argument of ParamSpec to be a string literal
Ts = TypeVarTuple(x)  # E: Expected first argument of TypeVarTuple to be a string literal
    "#,
);

testcase!(
    test_tvar_keyword_name,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar(name = "T")
P = ParamSpec(name = "P")
Ts = TypeVarTuple(name = "Ts")
    "#,
);

testcase!(
    test_tvar_unexpected_keyword,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar('T', foo=True)  # E: Unexpected keyword argument `foo`
P = ParamSpec('P', foo=True)  # E: Unexpected keyword argument `foo`
Ts = TypeVarTuple('Ts', foo=True)  # E: Unexpected keyword argument `foo`
    "#,
);

testcase!(
    test_tvar_kwargs,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar('T', **{'a': 'b'})  # E: Cannot pass unpacked keyword arguments to TypeVar
P = ParamSpec('P', **{'a': 'b'})  # E: Cannot pass unpacked keyword arguments to ParamSpec
Ts = TypeVarTuple('Ts', **{'a': 'b'})  # E: Cannot pass unpacked keyword arguments to TypeVarTuple
    "#,
);

testcase!(
    test_tvar_constraints_and_bound,
    r#"
from typing import TypeVar
T = TypeVar('T', int, bound=int)  # E: TypeVar cannot have both constraints and bound
    "#,
);

testcase!(
    test_tvar_variance,
    r#"
from typing import TypeVar
T1 = TypeVar('T1', covariant=True, contravariant=True)  # E: Contradictory variance specifications
T2 = TypeVar('T2', covariant=True, contravariant=False)
T3 = TypeVar('T3', covariant="lunch")  # E: Expected literal True or False
    "#,
);

testcase!(
    test_tvar_forward_ref,
    r#"
from typing import TypeVar
T1 = TypeVar('T1', bound='A')
T2 = TypeVar('T2', bound='B')  # E: Could not find name `B`
T3 = TypeVar('T3', 'A', int)
T4 = TypeVar('T4', 'B', int)  # E: Could not find name `B`
T5 = TypeVar('T5', default='A')
T6 = TypeVar('T6', default='B')  # E: Could not find name `B`

class A:
    pass
    "#,
);

testcase!(
    test_tvar_class_constraint,
    r#"
from typing import TypeVar
class A:
    pass
T1 = TypeVar('T1', int, A)
T2 = TypeVar('T2', int, B)  # E: Could not find name `B`
    "#,
);

testcase!(
    test_ordering_of_tparams_on_generic_base,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

class Base(Generic[T]):
    x: T

class Child(Base[S], Generic[T, S]):
    y: T

def f(c: Child[int, str]):
    assert_type(c.x, str)
    assert_type(c.y, int)
    "#,
);

testcase!(
    test_ordering_of_tparams_on_protocol_base,
    r#"
from typing import Protocol, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

class Base(Protocol[T]):
    x: T

class Child(Base[S], Protocol[T, S]):
    y: T

def f(c: Child[int, str]):
    assert_type(c.x, str)
    assert_type(c.y, int)
    "#,
);

testcase!(
    test_both_generic_and_protocol,
    r#"
from typing import Generic, Protocol, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
V = TypeVar("V")

class C(Protocol[V, T], Generic[S, T, U]):  # E: Class `C` specifies type parameters in both `Generic` and `Protocol` bases
    s: S
    t: T
    u: U
    v: V

def f(c: C[int, str, bool, bytes]):
    assert_type(c.s, int)
    assert_type(c.t, str)
    assert_type(c.u, bool)
    assert_type(c.v, bytes)
    "#,
);

testcase!(
    test_both_generic_and_implicit,
    r#"
from typing import Generic, Protocol, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

class C(Generic[T], list[S]):  # E: Class `C` uses type variables not specified in `Generic` or `Protocol` base
    t: T

def f(c: C[int, str]):
    assert_type(c.t, int)
    assert_type(c[0], str)
    "#,
);

testcase!(
    test_default,
    r#"
from typing import Generic, TypeVar, assert_type
T1 = TypeVar('T1')
T2 = TypeVar('T2', default=int)
class C(Generic[T1, T2]):
    pass
def f9(c1: C[int, str], c2: C[str]):
    assert_type(c1, C[int, str])
    assert_type(c2, C[str, int])
    "#,
);

testcase!(
    test_bad_default_order,
    r#"
from typing import Generic, TypeVar
T1 = TypeVar('T1', default=int)
T2 = TypeVar('T2')
class C(Generic[T1, T2]):  # E: Type parameter `T2` without a default cannot follow type parameter `T1` with a default
    pass
    "#,
);

testcase!(
    test_variance,
    r#"
from typing import Generic, TypeVar
T1 = TypeVar('T1', covariant=True)
T2 = TypeVar('T2', contravariant=True)
class C(Generic[T1, T2]):
    pass
class Parent:
    pass
class Child(Parent):
    pass
def f1(c: C[Parent, Child]):
    f2(c)  # E: Argument `C[Parent, Child]` is not assignable to parameter `c` with type `C[Child, Parent]`
def f2(c: C[Child, Parent]):
    f1(c)
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

fn env_exported_type_var() -> TestEnv {
    TestEnv::one(
        "lib",
        r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
"#,
    )
}

testcase!(
    test_imported,
    env_exported_type_var(),
    r#"
from lib import T

def f(x: T) -> T:
    y: T = x
    return y

x1: int = f(0)
x2: str = f("hello")
"#,
);

testcase!(
    bug = "TODO: We should raise an error on list[T] because T is unbounded",
    test_unbounded_typevar,
    r#"
from typing import TypeVar
T = TypeVar("T")
x: list[T]
    "#,
);

testcase!(
    test_typevar_violates_annotation,
    r#"
from typing import TypeVar
T: int = 0
T = TypeVar('T')  # E: `type[TypeVar(T, variance=PInvariant)]` is not assignable to variable `T` with type `int`
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
    test_generics,
    r#"
from typing import Literal
class C[T]: ...
def append[T](x: C[T], y: T):
    pass
v: C[int] = C()
append(v, "test")  # E: Argument `Literal['test']` is not assignable to parameter `y` with type `int`
"#,
);

testcase!(
    test_generics_legacy_unqualified,
    r#"
from typing import TypeVar, Generic
T = TypeVar("T")
class C(Generic[T]): ...
def append(x: C[T], y: T):
    pass
v: C[int] = C()
append(v, "test")  # E: Argument `Literal['test']` is not assignable to parameter `y` with type `int`
"#,
);

testcase!(
    test_generics_legacy_qualified,
    r#"
import typing
T = typing.TypeVar("T")
class C(typing.Generic[T]): ...
def append(x: C[T], y: T):
    pass
v: C[int] = C()
append(v, "test")  # E: Argument `Literal['test']` is not assignable to parameter `y` with type `int`
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
from typing import reveal_type
class A: ...
class B: ...
class C[T]: ...
class D[T = A]: ...
def f[E](e: type[E]) -> E: ...
reveal_type(f(A)) # E: revealed type: A
reveal_type(f(B)) # E: revealed type: B
reveal_type(f(C)) # E: revealed type: C[Unknown]
reveal_type(f(D)) # E: revealed type: D[A]
"#,
);

testcase!(
    bug = "This test is a placeholder, we've commented out the check for missing type arguments because until we have configurable errors it causes too many problems.",
    test_untype_with_missing_targs,
    r#"
class C[T]: pass

x: C        # TODO: The generic class `C` is missing type arguments.
y: C | int  # TODO: The generic class `C` is missing type arguments.
    "#,
);

testcase!(
    test_typevar_default_is_typevar_legacy,
    r#"
from typing import Generic, TypeVar, assert_type

T1 = TypeVar('T1', default=float)
T2 = TypeVar('T2', default=T1)

class A(Generic[T1, T2]):
    x: T2

def f(a: A[int]):
    assert_type(a.x, int)

def g(a: A):
    assert_type(a.x, float)
    "#,
);

testcase!(
    test_typevar_default_is_out_of_scope_typevar,
    r#"
from typing import Any, Generic, TypeVar, assert_type

T1 = TypeVar('T1')
T2 = TypeVar('T2', default=T1)
T3 = TypeVar('T3', default=T1 | T2)

class A(Generic[T2]):  # E: Default of type parameter `T2` refers to out-of-scope type parameter `T1`
    x: T2

class B(Generic[T3]):  # E: Default of type parameter `T3` refers to out-of-scope type parameters `T1`, `T2`
    pass

def f(a: A):
    assert_type(a.x, Any)
    "#,
);

testcase!(
    test_typevar_default_is_typevar,
    r#"
from typing import assert_type, TypeVar

class A[T1 = float, T2 = T1]: pass

T = TypeVar('T')
class B[S = T]: pass # E: out-of-scope type parameter `T`

def f(a1: A[int], a2: A):
    assert_type(a1, A[int, int])
    assert_type(a2, A[float, float])
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
from typing import reveal_type, Any
class A[*Ts, T = int]:  # E: TypeVar `T` with a default cannot follow TypeVarTuple `Ts`
    pass
class B[*Ts, T1, T2 = T1]:  # E: TypeVar `T2` with a default cannot follow TypeVarTuple `Ts`
    pass
reveal_type(B[int]()) # E: B[tuple[()], int, int]
reveal_type(B[int, str]()) # E: B[tuple[()], int, str]
reveal_type(B[int, str, float, bool, bytes]()) # E: B[tuple[int, str, float], bool, bytes]
# It doesn't matter too much how we fill in the type arguments when they aren't
# pinned by construction, as long as it's plausible.
reveal_type(B()) # E: revealed type: B[@_, @_, @_]
b: B[tuple[tuple[Any, ...], Any, Any]] = B()  # Here's one valid way to pin them
    "#,
);

testcase!(
    test_paramspec_with_default_after_typevartuple,
    r#"
from typing import Any, reveal_type
class A[*Ts, **P1, **P2 = P1]:
    pass
class B[*Ts, T, **P = [int, str]]:
    pass
reveal_type(A[[int, str]]()) # E: A[tuple[()], [int, str], [int, str]]
reveal_type(A[bool, [int, str]]()) # E: A[tuple[bool], [int, str], [int, str]]
reveal_type(A[bool, bytes, [int, str]]()) # E: A[tuple[bool, bytes], [int, str], [int, str]]
reveal_type(B[int, str, float]()) # E: B[tuple[int, str], float, [int, str]]
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
class A[*Ps, *Qs = *Ps]: # E: cannot be more than one TypeVarTuple
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
    assert_type(x, dict[Any, Any]) # should be dict[str, int]

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

testcase!(
    test_generic_with_type_checking_constant,
    r#"
import typing
if typing.TYPE_CHECKING: ...
T = typing.TypeVar('T')
class C(typing.Generic[T]):
    pass
    "#,
);

testcase!(
    test_error_on_bad_legacy_tparam,
    r#"
from typing import Any, Generic

# Explicit or implicit Any is not allowed.
class C1(Generic[Any]):  # E: Expected a type variable, got `Any`
    pass
def f() -> Any: ...
x = f()
class C2(Generic[x]):  # E: Expected a type variable, got `Unknown`
    pass

# But Any(Error) is.
T = oops()  # E:
class C3(Generic[T]):
    pass

class C4(Generic[int]):  # E: Expected a type variable, got `int`
    pass
    "#,
);
