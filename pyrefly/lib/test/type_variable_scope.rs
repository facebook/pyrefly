/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_tyvar_mix,
    r#"
from typing import TypeVar, assert_type
U = TypeVar("U")
def foo[T](
      x: U  # E: Type parameter `U` is not included in the type parameter list
    ) -> U:
    return x

assert_type(foo(1), int)
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
    test_init_class_scoped_typevars_in_self,
    r#"
from typing import Generic, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")

class Class8(Generic[T1, T2]):
    def __init__(self: "Class8[T2, T1]") -> None:  # E: `__init__` method self type cannot reference class type parameters `T2`, `T1`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
class Ok1(Generic[T]):
    def __init__(self: "Ok1[int]") -> None:
        pass
class Ok2[T]:
    def __init__(self: "Ok2[int]") -> None:
        pass
class Ok3(Generic[T]):
    def __init__(self) -> None:
        pass
class Ok4[T]:
    def __init__(self) -> None:
        pass
class Ok5(Generic[T]):
    def __init__[V](self: "Ok5[V]", arg: V) -> None:
        pass
class Ok6[T]:
    def __init__[V](self: "Ok6[V]", arg: V) -> None:
        pass
class Bad1(Generic[T]):
    def __init__(self: "Bad1[T]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
class Bad2[T]:
    def __init__(self: "Bad2[T]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope_nested,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
# Nested type variables should also be detected (e.g., Foo[list[T]])
class Bad1(Generic[T]):
    def __init__(self: "Bad1[list[T]]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
class Bad2[T]:
    def __init__(self: "Bad2[tuple[T, int]]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope_overload,
    r#"
from typing import Generic, TypeVar, overload
T = TypeVar("T")
# Overloaded __init__ methods should also be checked
class Bad1(Generic[T]):
    @overload
    def __init__(self: "Bad1[T]", x: int) -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        ...
    @overload
    def __init__(self: "Bad1[str]", x: str) -> None:
        ...
    def __init__(self, x: int | str) -> None:
        pass
class Ok1(Generic[T]):
    @overload
    def __init__(self: "Ok1[int]", x: int) -> None:
        ...
    @overload
    def __init__(self: "Ok1[str]", x: str) -> None:
        ...
    def __init__(self, x: int | str) -> None:
        pass
"#,
);

testcase!(
    test_class_scoped_typevar_in_decorated_init,
    r#"
from typing import Any
def decorate(f) -> Any: ...
class A[T]:
    @decorate
    def __init__(self: A[T]): ...  # E: self type cannot reference class type parameter `T`
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
    test_out_of_scope_old_typevar,
    r#"
from typing import Any, Callable, TypeVar
T = TypeVar('T')
def f() -> Any: ...
def g():
    x: T = f()  # E: Type variable `T` is not in scope
def h() -> Callable[[T], T]:
    # T appears in the return type, so LegacyTParamCollector treats it as
    # a type parameter of h. This matches pyright's behavior.
    x: T = f()
    return lambda x: x
    "#,
);

testcase!(
    test_unbounded_typevar,
    r#"
from typing import TypeVar
T = TypeVar("T")
x: list[T]  # E: Type variable `T` is not in scope
    "#,
);

// Because the scoped versions of legacy tparams are a static-only concept
// but scope is well-defined runtime concept, we wind up with weird edge cases
// where Pyrefly's scope can do the wrong thing.
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
    test_multiple_possible_legacy_tparams,
    TestEnv::one(
        "foo",
        "from typing import TypeVar\nT = TypeVar('T')\nclass C: pass"
    ),
    r#"
from typing import Generic, assert_type
import foo

# `foo.T` and `foo.C` are both hosted on the `foo` module and collapse onto a single
# base-name scope entry. We narrow `foo` at every hosted legacy type parameter's facet
# (not just the last one added), so references to each resolve to the right Quantified.
def f(x: foo.T, y: foo.C) -> foo.T:
    z: foo.T = x
    return z
assert_type(f(1, foo.C()), int)

class MyList(Generic[foo.T], list[tuple[foo.C, foo.T]]):
    def my_append(self, c: foo.C, t: foo.T):
        self.append((c, t))
my_list: MyList[int] = MyList()
my_list.my_append(foo.C(), 5)
    "#,
);

// Each hosted legacy tparam is reported at its own range, not all at the range of the last one.
testcase!(
    test_multiple_possible_legacy_tparams_with_scoped_tparams,
    TestEnv::one(
        "foo",
        "from typing import TypeVar\nT = TypeVar('T')\nS = TypeVar('S')"
    ),
    r#"
import foo

def f[X](
    a: X,
    b: foo.T,  # E: Type parameter `T` is not included in the type parameter list
    c: foo.S,  # E: Type parameter `S` is not included in the type parameter list
) -> None: ...
    "#,
);

testcase!(
    test_recover_gracefully_from_out_of_scope_typevartuple,
    r#"
from typing import TypeVarTuple, reveal_type
Ts = TypeVarTuple("Ts")
class E[R, T = tuple[*Ts]]: ...   # E: out-of-scope type parameter `Ts`
def f(x: E[str]):
    reveal_type(x)  # E: E[str, tuple[Unknown, ...]]
    "#,
);

testcase!(
    test_typevar_scoping_restrictions,
    r#"
from typing import TypeVar, Generic, TypeAlias
from collections.abc import Iterable

T = TypeVar("T")
S = TypeVar("S")

# Unbound TypeVar S used in generic function body
def fun_3(x: T) -> list[T]:
    y: list[T] = []  # OK
    z: list[S] = []  # E: Type variable `S` is not in scope
    return y

# Unbound TypeVar S in class body (not in method)
class Bar(Generic[T]):
    an_attr: list[S] = []  # E: Type variable `S` is not in scope

# Nested class using outer class's TypeVar
class Outer(Generic[T]):
    class Bad(Iterable[T]):  # E: shadows
        ...
    class AlsoBad:
        x: list[T]  # E: Type variable `T` is not in scope

    alias: TypeAlias = list[T]  # E: Type variable `T` is not in scope

# Unbound TypeVars at global scope
global_var1: T  # E: Type variable `T` is not in scope
global_var2: list[T] = []  # E: Type variable `T` is not in scope
list[T]()  # E: Type variable `T` is not in scope
"#,
);

testcase!(
    test_nested_class_independent_typevar_adoption,
    r#"
from typing import Generic, Type, TypeVar

_Deserialized = TypeVar("_Deserialized")
_Serialized = TypeVar("_Serialized")

class CustomCoercer(Generic[_Deserialized, _Serialized]):
    # CoercerMapping uses the same TypeVars as CustomCoercer, which the spec forbids.
    class CoercerMapping(
        dict[
            Type[_Deserialized],  # E: shadows
            Type["CustomCoercer[_Deserialized, _Serialized]"],  # E: shadows
        ]
    ):
        # The method binds new type parameters, but may not reuse the enclosing names.
        def __getitem__(
            self,
            key: type[_Deserialized],  # E: shadows
        ) -> type["CustomCoercer[_Deserialized, _Serialized]"]: ...  # E: shadows
"#,
);

testcase!(
    test_nested_class_outer_legacy_tparam_out_of_scope,
    r#"
from typing import Generic, TypeVar
from collections.abc import Iterable

T = TypeVar("T")
S = TypeVar("S")

class Outer(Generic[T]):
    # A method of the enclosing class may use its type parameter.
    def m(self, x: T) -> T:
        return x

    # A nested class does not inherit the enclosing class's type parameters, so `T` is out of
    # scope in its base list and its body.
    class Bad(Iterable[T]):  # E: shadows
        ...

    class AlsoBad:
        x: list[T]  # E: Type variable `T` is not in scope

        # The method binds its own `T`, which illegally shadows the enclosing parameter.
        def method(self, y: T) -> None:  # E: shadows
            ...

    # A nested class may still introduce its own, independent type parameter.
    class Inner(Iterable[S]):
        ...
"#,
);

testcase!(
    test_type_alias_cannot_capture_enclosing_tparam,
    r#"
from typing import Generic, TypeVar, TypeAlias

T = TypeVar("T")

# A generic type alias at module scope is fine; the free TypeVar parametrizes the alias.
ModuleAlias: TypeAlias = list[T]  # OK

class Outer(Generic[T]):
    # A type alias defined in a class body may not capture the class's type parameter.
    explicit: TypeAlias = list[T]  # E: Type variable `T` is not in scope
    implicit = dict[str, T]  # E: Type variable `T` is not in scope

    # A method may still use the class's type parameter.
    def m(self, x: T) -> T:
        return x
"#,
);

testcase!(
    test_out_of_scope_typevar_in_expression,
    r#"
from typing import Generic, TypeVar

T = TypeVar("T")

# Instantiating a generic subscripted with an out-of-scope TypeVar is an error.
list[T]()  # E: Type variable `T` is not in scope

# But a TypeVar may appear as a value (it is a runtime object), and a subscripted generic
# may be used as an implicit type alias.
x = T  # OK
alias = list[T]  # OK

def f(a: T) -> T:
    # Inside a generic function the TypeVar is in scope.
    b = list[T]()  # OK
    return a

class C(Generic[T]):
    def m(self) -> None:
        list[T]()  # OK: T is in scope in a method of the generic class
"#,
);

testcase!(
    test_class_typevar_shadowing_enclosing_type_var,
    r#"
from typing import Generic, TypeVar

T = TypeVar("T")

class A(Generic[T]):
    class B(Generic[T]): ...  # E: shadows

def f(x: T) -> T:
    class C(Generic[T]): ...  # E: shadows
    return x

class D[T]:
    class E[T]: ...  # E: shadows

def g[T](x: T) -> T:
    class F[T]: ...  # E: shadows
    return x
    "#,
);

testcase!(
    test_function_typevar_shadowing_enclosing_typevar,
    r#"
from typing import Generic, TypeVar, reveal_type

T = TypeVar("T")

class A(Generic[T]):
    class B:
        # This `T` is not allowed to refer to `A.T` from the outer class scope, so it must be
        # function-scoped.
        def f(self, x: T) -> T: ...  # E: shadows

class C[T]:
    class D:
        def f[T](self, x: T) -> T: ...  # E: shadows

# We don't allow shadowing because it's confusing, but we resolve the signatures correctly regardless.
reveal_type(A.B.f)  # E: [T](self: A.B, x: T) -> T
reveal_type(C.D.f)  # E: [T](self: C.D, x: T) -> T
    "#,
);

testcase!(
    test_outer_class_typevar_is_out_of_scope_in_default_and_body,
    r#"
from typing import Any, Generic, TypeVar, assert_type

LegacyOuter = TypeVar("LegacyOuter")
LegacyInner = TypeVar("LegacyInner", default=LegacyOuter)

class A(Generic[LegacyOuter]):
    class B(Generic[LegacyInner]):  # E: refers to out-of-scope type parameter
        x: LegacyOuter  # E: not in scope
        y: LegacyInner

assert_type(A.B().x, Any)
assert_type(A.B().y, Any)

class C[Outer]:
    class D[Inner = Outer]:  # E: not in scope
        x: Outer  # E: not in scope
        y: Inner

assert_type(C.D().x, Any)
assert_type(C.D().y, Any)
    "#,
);

testcase!(
    test_outer_class_typevar_is_out_of_scope_in_bases,
    r#"
from typing import Generic, TypeVar

LegacyT = TypeVar("LegacyT")

class A(Generic[LegacyT]):
    class B(list[LegacyT]):  # E: shadows
        pass

class C[T]:
    class D(list[T]):  # E: not in scope
        pass
    "#,
);

testcase!(
    test_outer_function_typevar_is_in_scope,
    r#"
from typing import Generic, TypeVar, assert_type

LegacyT = TypeVar("LegacyT")

def f(x: LegacyT):
    class A:
        x: LegacyT
    return A.x

def g[T](x: T):
    class A:
        x: T
    return A.x

assert_type(f(0), int)
assert_type(g(0), int)
    "#,
);

testcase!(
    test_cannot_redeclare_outer_typevar_in_class_in_method,
    r#"
from typing import Generic, TypeVar, assert_type
T = TypeVar("T")
class A[T]:
    def f1(self):
        class C(list[T]): ...  # E: not in scope
    def f2(self):
        class C[T](list[T]): ...  # E: shadows
class B(Generic[T]):
    def f1(self):
        class C(list[T]): ...  # E: shadows
    def f2(self):
        class C[T](list[T]): ...  # E: shadows
    "#,
);

testcase!(
    test_inner_method_can_access_outer_typevar_as_value,
    r#"
from typing import Generic, TypeVar, assert_type
class Outer[T]:
    class Inner:
        def f(self):
            assert_type(T, TypeVar)
LegacyT = TypeVar("LegacyT")
class LegacyOuter(Generic[LegacyT]):
    class Inner:
        def f(self):
            assert_type(LegacyT, TypeVar)
    "#,
);

testcase!(
    test_shadowing_detected_with_intervening_decl,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
class A(Generic[T]):
    T: int  # This `T` should not prevent us from detecting that `B.T` shadows `A.T`
    class B(Generic[T]): ...  # E: shadows
    "#,
);

testcase!(
    test_illegally_shadowing_class_is_still_generic,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
def f(x: T):
    class C(Generic[T]): ...  # E: shadows
    # Even though C.T illegally shadows f.T, we still gracefully recover and
    # treat C as a generic class.
    c1: C[int] = C[int]()
    c2: C[str] = C[int]()  # E: `f.C[int]` is not assignable to `f.C[str]`
    "#,
);
