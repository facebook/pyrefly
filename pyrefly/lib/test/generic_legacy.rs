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
    bug = "We allow annotated legacy type vars as *values*, but they don't work downstream.",
    test_annotated_legacy_type_var,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple
T_bad_ann: int = TypeVar("T_bad_ann")  # E: `TypeVar` is not assignable to `int`
P_bad_ann: int = ParamSpec("P_bad_ann")  # E: `ParamSpec` is not assignable to `int`
Ts_bad_ann: int = TypeVarTuple("Ts_bad_ann")  # E: `TypeVarTuple` is not assignable to `int`

# All of these are legal assignments, but we won't understand them downstream
T: TypeVar = TypeVar("T")
P: ParamSpec = ParamSpec("P")
Ts: TypeVarTuple = TypeVarTuple("Ts")

# For example, see this:
def f(x: T) -> T:  # E: Expected a type form, got instance of `TypeVar  # E: Expected a type form, got instance of `TypeVar`
    pass
"#,
);

testcase!(
    test_typevar_values,
    r#"
from typing import TypeVar, ParamSpec, TypeVarTuple, Callable, assert_type

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")

assert_type(T, TypeVar)
assert_type(P, ParamSpec)
assert_type(Ts, TypeVarTuple)

def f(x: T, xs: tuple[*Ts], f: Callable[P, None]):
    assert_type(T, TypeVar)
    assert_type(P, ParamSpec)
    assert_type(Ts, TypeVarTuple)

    assert_type(x, T)
    assert_type(xs, tuple[*Ts])
    assert_type(f, Callable[P, None])

def g[U, *Us, **Q](x: U, xs: tuple[*Us], f: Callable[Q, None]):
    assert_type(U, TypeVar)
    assert_type(Q, ParamSpec)
    assert_type(Us, TypeVarTuple)

    assert_type(x, U)
    assert_type(xs, tuple[*Us])
    assert_type(f, Callable[Q, None])
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
    test_legacy_generic_syntax_duplicated_names,
    r#"
from typing import Any, Generic, Protocol, TypeVar, TypeVarTuple, ParamSpec
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
P = ParamSpec('P')

class A(Generic[T, T]):  # E: Duplicated type parameter declaration
    pass
class B(Generic[*Ts, *Ts]):  # E: Duplicated type parameter declaration
    pass
class C(Generic[P, P]):  # E: Duplicated type parameter declaration
    pass

class D(Protocol[T, T]):  # E: Duplicated type parameter declaration
    pass
class E(Protocol[*Ts, *Ts]):  # E: Duplicated type parameter declaration
    pass
class F(Protocol[P, P]):  # E: Duplicated type parameter declaration
    pass
    "#,
);

testcase!(
    test_legacy_generic_syntax_implicit_targs,
    TestEnv::new().enable_implicit_any_error(),
    r#"
from typing import Any, Generic, TypeVar, assert_type
T = TypeVar('T')
class A(Generic[T]):
    x: T
def f(a: A):  # E: Cannot determine the type parameter `T` for generic class `A`
    assert_type(a.x, Any)
    "#,
);

testcase!(
    test_legacy_generic_syntax_implicit_targs_with_default,
    TestEnv::new().enable_implicit_any_error(),
    r#"
from typing import Any, Generic, TypeVar, assert_type
T = TypeVar('T')
U = TypeVar('U', default=int)
class A(Generic[T, U]):
    x: T
    y: U
def f(a: A):  # E: Cannot determine the type parameter `T` for generic class `A`
    assert_type(a.x, Any)
    assert_type(a.y, int)
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
T3 = TypeVar('T3', covariant="lunch")  # E: Expected literal `True` or `False`
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

testcase!(
    test_legacy_typevar_revealed_type,
    r#"
from typing import reveal_type, TypeVar

T = TypeVar("T")
TypeForm = type[T]

reveal_type(T)  # E: TypeVar[T]
reveal_type(TypeForm)  # E: revealed type: type[type[T]]
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
    test_typevar_default_is_legacy_typevar,
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
    test_scoped_typevar_default_is_legacy_typevar,
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
    bug = "We should error on out-of-scope typevars",
    test_out_of_scope_old_typevar,
    r#"
from typing import Any, Callable, TypeVar
T = TypeVar('T')
def f() -> Any: ...
def g():
    x: T = f()  # this should be an error
def h() -> Callable[[T], T]:
    # This should be an error. Note that we treat `[T]() -> ((T) -> T)` as `() -> ([T](T) -> T)`,
    # which makes `T` out-of-scope in the body.
    x: T = f()
    return lambda x: x
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
T = TypeVar('T')  # E: `TypeVar[T]` is not assignable to variable `T` with type `int`
    "#,
);

testcase!(
    test_function_legacy_typevar_dotted_name,
    env_exported_type_var(),
    r#"
import lib
from typing import assert_type

def f(x: lib.T) -> lib.T:
    return x
assert_type(f(0), int)
    "#,
);

testcase!(
    test_class_legacy_typevar_dotted_name,
    env_exported_type_var(),
    r#"
import lib
from typing import assert_type, Generic

class A(Generic[lib.T]):
    x: lib.T
assert_type(A[int]().x, int)
    "#,
);

testcase!(
    test_legacy_typevar_defined_after_use,
    r#"
from __future__ import annotations
from typing import TypeVar

class Session:
    def __enter__(self: _S) -> _S:
        return self
    def __exit__(self, type_, value, traceback):
        pass
    def begin(self):
        pass

_S = TypeVar("_S", bound="Session")

with Session() as session:
    session.begin()
    "#,
);

testcase!(
    test_legacy_typevar_imported_after_use,
    TestEnv::one("foo", "from typing import TypeVar\nT = TypeVar('T')"),
    r#"
from typing import assert_type
def f(x: "foo.T") -> "foo.T":
    return x
import foo
assert_type(f(0), int)
    "#,
);

// This test case is needed to avoid a regression resolving special binding-time
// information that travels through a legacy tparam builder.
//
// It is necessary because Pyrefly sees the `bool` in a type annotation and has
// to account for the possiblity that `bool` (which is an import from builtins)
// might actually be a legacy type variable.
//
// We have to make sure that the way we do this doesn't break special export
// lookups in the binding code; this test guards against regressions.
testcase!(
    test_bool_special_exports_bug,
    r#"
from typing import assert_type, Literal
def f(x: bool):
    if bool(x):
        assert_type(x, Literal[True])
    else:
        assert_type(x, Literal[False])
    "#,
);

// Because the scoped versions of legacy tparams are a static-only concept
// but scope is well-defined runtime concept, we wind up with weird edge cases
// where Pyrefy's scope can do the wrong thing.
//
// One thing to watch out for is that it would be a false positive if reading
// a possible legacy tparam as a value triggers an uninitialized local error.
//
// This came up in a refactor and was only caught by end-to-end pydantic tests;
// this unit test checks against a regression.
testcase!(
    test_possible_legacy_tparams_used_as_values,
    TestEnv::one("foo", "class A: pass"),
    r#"
from foo import A
class C[T]: pass
class D(C[A]):
    x: A = A()
"#,
);

testcase!(
    bug = "We model tparam intercepts in static scope, which shadows parents and can lead to edge case bugs involving mutable captures",
    test_shadowing_interaction_with_mutable_capture,
    r#"
def f(x: A):
    nonlocal A  # Should error, but it finds the annotation scope with a fake entry for `A` as a potential tparam.

class A:
    pass
    "#,
);

testcase!(
    bug = "We currently create separate narrows for modules that may contain legacy type variables, we need to merge them",
    test_mutliple_possible_legacy_tparams,
    TestEnv::one(
        "foo",
        "from typing import TypeVar\nT = TypeVar('T')\nclass C: pass"
    ),
    r#"
from typing import Generic, assert_type
import foo

# Here, the `foo.C` possible-legacy-tparam binding is the one that winds up in scope, we
# lose track of the `foo.T` one. It probably doesn't matter very much since we at least
# understand the signature correctly.
def f(x: foo.T, y: foo.C) -> foo.T:
    z: foo.T = x  # E: `T` is not assignable to `TypeVar[T]`
    return z  # E: Returned type `TypeVar[T]` is not assignable to declared return type `T
assert_type(f(1, foo.C()), int)

# The same thing happens here, but it's a much bigger problem because now we forget
# about the type variable identity for the entire class body, so the signatures come out
# wrong.
class MyList(Generic[foo.T], list[tuple[foo.C, foo.T]]):
    def my_append(self, c: foo.C, t: foo.T):
        self.append((c, t))  # E: Argument `tuple[C, TypeVar[T]]` is not assignable to parameter `object` with type `tuple[C, T]` in function `list.append`
my_list: MyList[int] = MyList()
my_list.my_append(foo.C(), 5)  # E: Argument `Literal[5]` is not assignable to parameter `t` with type `TypeVar[T]` in function `MyList.my_append`
    "#,
);
