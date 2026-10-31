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
    test_metaclass_instance_not_subtype_of_specific_type,
    r#"
from typing import Any
class Meta(type): pass
class A(metaclass=Meta): pass

def f(x: Meta):
    a: type[A] = x  # E: `Meta` is not assignable to `type[A]`
    any_cls: type[Any] = x
    o: type[object] = x
    union_with_object: type[int | object] = x
    union_without_object: type[int | str] = x  # E: `Meta` is not assignable to `type[int | str]`
"#,
);

testcase!(
    test_concrete_metaclass_class_object_subtyping,
    r#"
class Meta(type): pass
class Base(metaclass=Meta):
    def only_on_base(self) -> str:
        return "base"
class Child(Base): pass

def needs_base_class(cls: type[Base]) -> str:
    return cls().only_on_base()

def make[T: Base](cls: type[T]) -> T:
    return cls()

cls = Child
needs_base_class(Child)
needs_base_class(cls)
child: Child = make(Child)

def f(x: Meta):
    unsafe: type[Base] = x  # E: `Meta` is not assignable to `type[Base]`
    make(x)  # E: Argument `Meta` is not assignable to parameter `cls`
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
    x: type[int] = a  # E: `A` is not assignable to `type[int]`
    y: type[object] = a
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
    test_type_function_in_base_class_list_v0,
    r#"
class A:
    pass
a = A()
class B(type(a)):
    pass
    "#,
);

testcase!(
    test_type_function_in_base_class_list_v1,
    r#"
class A:
    pass
class B(type(A)):
    pass
    "#,
);

testcase!(
    test_type_function_in_base_class_list_v2,
    r#"
from typing import assert_type, ClassVar, Any
class M(type):
    x: ClassVar[int] = 42
class A(metaclass=M):
    pass
class B(type(A)):
    pass
assert_type(B.x, int)
    "#,
);

testcase!(
    test_call_returning_type_in_base_class_list,
    r#"
from typing import Any, assert_type

class Mixin:
    def greet(self) -> str:
        return "hi"

def factory(name: str) -> type[Mixin]: ...
def untyped_factory(name: str) -> Any: ...

class C(factory("x")):
    pass

class D(untyped_factory("x")):
    pass

assert_type(C().greet(), str)
    "#,
);

testcase!(
    test_call_base_legacy_typevar_scoping,
    r#"
from typing import Any, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

def ident(x: type[S]) -> type[S]: ...

class C(ident(list[T])):  # E: Type variable `T` is not in scope
    pass

assert_type(C()[0], Any)

class D(ident(list[int])):
    pass

assert_type(D()[0], int)
    "#,
);

testcase!(
    test_call_base_bound_legacy_typevar,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")
S = TypeVar("S")

def ident(x: type[S]) -> type[S]: ...

class C(ident(list[T]), Generic[T]):
    pass

assert_type(C[int]()[0], int)
assert_type(C[str]()[0], str)
    "#,
);

testcase!(
    test_call_base_imported_legacy_typevars,
    TestEnv::one(
        "params",
        r#"
from typing import TypeVar
T = TypeVar("T")
"#,
    ),
    r#"
from typing import Any, Generic, TypeVar, assert_type
import params

T = TypeVar("T")
S = TypeVar("S")

def ident(x: type[S]) -> type[S]: ...

class C(ident(list[params.T]), Generic[params.T]):
    pass

assert_type(C[int]()[0], int)

class D(ident(list[T]), Generic[params.T]):  # E: Type variable `T` is not in scope
    pass

assert_type(D[int]()[0], Any)

class Factory(Generic[params.T]):
    def __new__(cls) -> type[list[params.T]]: ...

class E(Factory[params.T](), Generic[params.T]): ...  # E: Type variable `T` is not in scope
assert_type(E[int]()[0], Any)
    "#,
);

testcase!(
    test_call_base_argument_validation,
    r#"
def factory(name: str) -> type[int]:
    return int

class C(factory(1)):  # E: Argument `Literal[1]` is not assignable to parameter `name` with type `str`
    pass
    "#,
);

testcase!(
    bug = "Constructor callees erase class-bound legacy TypeVars",
    test_call_base_constructor_legacy_typevar,
    r#"
from typing import Any, Generic, TypeVar, assert_type

T = TypeVar("T")

class Factory(Generic[T]):
    def __new__(cls) -> type[list[T]]: ...

class C(Factory[T](), Generic[T]):  # E: Type variable `T` is not in scope
    pass

assert_type(C[int]()[0], Any)

def ident[S](x: type[S]) -> type[S]: ...
class D(ident(Factory[T]()), Generic[T]): ...  # E: Type variable `T` is not in scope
assert_type(D[str]()[0], Any)
class E(type(Factory[T]()), Generic[T]): ...  # E: Type variable `T` is not in scope
assert_type(E[int]()[0], Any)
    "#,
);

testcase!(
    test_call_base_constructor_pep695,
    r#"
from typing import assert_type

class Factory[T]:
    def __new__(cls) -> type[list[T]]: ...

class C[T](Factory[T]()):
    pass

assert_type(C[int]()[0], int)
assert_type(C[str]()[0], str)
    "#,
);

testcase!(
    bug = "Constructor callees erase class-bound ParamSpecs",
    test_call_base_constructor_paramspec,
    r#"
from typing import Generic, ParamSpec, assert_type

P = ParamSpec("P")
class Callback(Generic[P]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> int: ...

class Factory(Generic[P]):
    def __new__(cls) -> type[Callback[P]]: ...

class C(Factory[P](), Generic[P]): ...  # E: Type variable `P` is not in scope

def check(x: C[[str, int]]) -> None:
    assert_type(x.run("x", 1), int)
    x.run(1, 1)
    "#,
);

testcase!(
    bug = "Constructor callees erase class-bound TypeVarTuples",
    test_call_base_constructor_typevartuple,
    r#"
from typing import Generic, TypeVarTuple, Unpack, assert_type

Ts = TypeVarTuple("Ts")
class Factory(Generic[Unpack[Ts]]):
    def __new__(cls) -> type[tuple[Unpack[Ts]]]: ...

class C(Factory[Unpack[Ts]](), Generic[Unpack[Ts]]): ...  # E: Type variable `Ts` is not in scope
assert_type(C[int, str]()[0], object)
assert_type(C[int, str]()[1], object)
    "#,
);

testcase!(
    bug = "Constructor type arguments and runtime values need separate scoping",
    test_call_base_constructor_runtime_typevar,
    r#"
from typing import Any, Generic, TypeVar, assert_type

T = TypeVar("T")
class Factory(Generic[T]):
    def __new__(cls, tv: TypeVar) -> type[list[T]]: ...

class C(Factory[T](T), Generic[T]): ...  # E: Type variable `T` is not in scope
assert_type(C[str]()[0], Any)
    "#,
);

testcase!(
    test_call_base_binding_order,
    r#"
from typing import assert_type

class Mixin:
    def greet(self) -> str: ...
class Meta(type): ...

def factory(meta: type) -> type[Mixin]: ...
class C(factory(M := Meta), metaclass=M): pass
assert_type(C().greet(), str)

class Parent: ...
def save(base: type) -> type[Mixin]: ...
class D(save(B := Parent), B): pass
assert_type(D().greet(), str)
    "#,
);

testcase!(
    test_call_base_lambda_and_comprehension_shadowing,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")
def factory(x: type[int]) -> type[list[int]]: ...

# Local values named T must not be rebound to the class's type parameter.
class Lambda(factory((lambda T: T)(int)), Generic[T]): pass
assert_type(Lambda[str]()[0], int)

class Comprehension(factory([T for T in [int]][0]), Generic[T]): pass
assert_type(Comprehension[str]()[0], int)
    "#,
);

testcase!(
    test_call_base_class_dependencies,
    r#"
from typing import Any, assert_type

class Mixin:
    def greet(self) -> str: ...

# Argument checking depends on the metadata of the class being defined.
def factory(x: "C | int") -> type[Mixin]: ...

class C(factory(1)): pass
assert_type(C().greet(), str)

# Return inference depends on the class being defined. A union of two
# distinct classes is still invalid, even when inference encounters a cycle.
def inferred_factory(flag: bool):
    if flag:
        return Mixin
    return D

class D(inferred_factory(True)): ...  # E: Invalid base class:
assert_type(D().greet(), Any)
    "#,
);

testcase!(
    test_call_base_generic_factories,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T", bound=int)
S = TypeVar("S", int, str)

def ident[U](x: type[U]) -> type[U]: ...

class Bounded(ident(list[T]), Generic[T]): pass
assert_type(Bounded[int]()[0], int)

class Constrained(ident(list[S]), Generic[S]): pass
assert_type(Constrained[str]()[0], str)

class Pep695[T](ident(list[T])): pass
assert_type(Pep695[int]()[0], int)

def generic_outer[T](x: T):
    class Inner(ident(list[T])): pass
    assert_type(Inner()[0], T)
    "#,
);

testcase!(
    test_call_base_overloaded_callable_and_wrapped_factories,
    r#"
from typing import Literal, assert_type, cast, overload

class Mixin:
    def greet(self) -> str: ...

class Other: ...

@overload
def select(x: Literal[True]) -> type[Mixin]: ...
@overload
def select(x: Literal[False]) -> type[Other]: ...
def select(x: bool) -> type[Mixin] | type[Other]: ...

class Selected(select(True)): pass
assert_type(Selected().greet(), str)

class CallableFactory:
    def __call__(self) -> type[Mixin]: ...

make = CallableFactory()
class CallableChild(make()): pass
assert_type(CallableChild().greet(), str)

class Wrapped(cast(type[Mixin], Mixin)): pass
assert_type(Wrapped().greet(), str)
    "#,
);

testcase!(
    test_call_base_invalid_returns,
    r#"
from typing import Literal, Never

class Mixin: ...
class Other: ...

def instance_factory() -> Mixin: ...
def none_factory() -> None: ...
def never_factory() -> Never: ...
def union_factory() -> type[Mixin] | type[Other]: ...
def literal_factory() -> Literal[1]: ...

class Instance(instance_factory()): ...  # E: Expected a type form, got instance of `Mixin`  # E: Invalid base class: `Mixin`
class Null(none_factory()): ...  # E: Invalid base class: `None`
class Bottom(never_factory()): ...  # E: Expected a type form, got instance of `Never`  # E: Invalid base class: `Never`
class Union(union_factory()): ...  # E: Invalid base class: `Mixin | Other`
class Lit(literal_factory()): ...  # E: Expected a type form, got instance of `Literal[1]`  # E: Invalid base class: `Literal[1]`
    "#,
);

testcase!(
    test_call_base_final_and_protocol,
    r#"
from typing import Protocol, assert_type, final

@final
class Final: ...

def final_factory() -> type[Final]: ...
class FinalChild(final_factory()): ...  # E: Cannot extend final class `Final`

class P(Protocol):
    def greet(self) -> str: ...

def protocol_factory() -> type[P]: ...
class Concrete(protocol_factory()):
    def greet(self) -> str: return "hi"

assert_type(Concrete().greet(), str)
    "#,
);

testcase!(
    test_call_base_runtime_typevar_arguments,
    r#"
from typing import Generic, TypeVar, assert_type

T = TypeVar("T")
class Mixin:
    def greet(self) -> str: ...

def factory(x: TypeVar) -> type[Mixin]: ...

# A TypeVar passed as a runtime value is not a type argument.
class C(factory(T), Generic[T]): pass
assert_type(C[int]().greet(), str)
    "#,
);

testcase!(
    test_type_function_in_base_class_list_self_cycle,
    r#"
from typing import assert_type
class B(type(B)):
    pass
# B references itself in its base class via type(B).
# The cycle is detected and resolved: type(B) degrades
# to `type` (the default metaclass), so B inherits from `type`.
x: type = B
    "#,
);

testcase!(
    test_type_function_in_base_class_list_mutual_cycle,
    r#"
class C(type(D)):
    pass
class D(type(C)):
    pass
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
    test_conditional_import_generic_base_class,
    r#"
from typing import Any, assert_type

class Base[T]:
    def get(self) -> T: ...

DynamicBase: type[Base[int]] | Any = Base[int]

class Child(DynamicBase):
    pass

assert_type(Child().get(), int)
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

testcase!(
    test_unknown_base_class,
    TestEnv::new().enable_unknown_base_class_error(),
    r#"
from typing import Any, NewType

class Known: ...

def ident(x):
    return x

def f(cls: type, cls2, cls3: type[Any], c: bool):
    class A(cls): ...  # E: Type of base class is Unknown
    class B(cls2): ...  # E: Type of base class is Unknown
    class C(Known, cls2): ...  # E: Type of base class is Unknown
    class D(cls3): ...
    class D2(cls2.attr): ...  # E: Type of base class is Unknown
    # A union of a known class and an implicit `Any` is not reported.
    Partial = Known if c else cls2
    class E(Partial): ...
    # NewType reports its own error instead.
    N = NewType("N", cls2)  # E: Second argument to NewType is invalid

Untyped = ident(Known)
class F(Untyped): ...  # E: Type of base class is Unknown
class G(Any): ...

# Reading a variable annotated with explicit `Any` produces an implicit `Any`.
X: Any = Known
class H(X): ...  # E: Type of base class is Unknown

Dynamic: type[Any] = Known
class I(Dynamic): ...
"#,
);

testcase!(
    test_unknown_base_class_error_recovery_any_not_reported,
    TestEnv::new().enable_unknown_base_class_error(),
    r#"
from missing import M  # E: Cannot find module `missing`
class A(M): ...
class B(Undefined): ...  # E: Could not find name `Undefined`
"#,
);

testcase!(
    test_unknown_base_class_ignored_by_default,
    r#"
def f(cls):
    class A(cls): ...
"#,
);
