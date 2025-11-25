/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_type_alias_simple,
    r#"
from typing import assert_type
type X = int
def f(x: X):
    assert_type(x, int)
    "#,
);

testcase!(
    test_type_alias_type,
    r#"
from typing import assert_type, TypeAliasType, TypeVar
X1 = TypeAliasType("X1", int)
def f(x: X1):
    assert_type(x, int)

T = TypeVar('T')
T2 = TypeVar('T2')
X2 = TypeAliasType("X2", list[T], type_params=(T,))
X3 = TypeAliasType("X3", list[T])  # E: Type variable `T` is out of scope for this `TypeAliasType`
X4 = TypeAliasType("X4", list[T] | list[T2], type_params=(T,))  # E: Type variable `T2` is out of scope for this `TypeAliasType`
X5 = TypeAliasType(name="X5", value=int)

def f2(x: X2[int]):
    assert_type(x, list[int])
    "#,
);

testcase!(
    test_type_alias_generic,
    r#"
from typing import assert_type
type X[T] = list[T]
def f(x: X[int]):
    assert_type(x, list[int])
    "#,
);

testcase!(
    test_type_alias_missing_quantifieds,
    r#"
from typing import TypeVar
T = TypeVar('T')
type X = list[T]  # E: Type parameters used in `X`
    "#,
);

testcase!(
    test_type_alias_unused_quantifieds,
    r#"
# Questionable code, but not an error
type X[T] = list
    "#,
);

testcase!(
    test_bad_type_alias,
    r#"
type X = 1  # E: number literal cannot be used in annotations
    "#,
);

testcase!(
    test_generic_alias_implicit,
    r#"
from typing import TypeVar, assert_type
T = TypeVar('T')
X = list[T]
def f(x: X[int]):
    assert_type(x, list[int])
    "#,
);

testcase!(
    test_any_alias_implicit,
    r#"
from typing import Any, TypeAlias, assert_type
T = Any
def f(x: T):
    assert_type(x, Any)
f(0)
f(None)
"#,
);

testcase!(
    test_any_alias_explicit,
    r#"
from typing import Any, TypeAlias, assert_type
T: TypeAlias = Any
def f(x: T):
    assert_type(x, Any)
f(0)
f(None)
"#,
);

testcase!(
    test_generic_alias_explicit,
    r#"
from typing import TypeAlias, TypeVar, assert_type
T = TypeVar('T')
X: TypeAlias = list[T]
def f(x: X[int]):
    assert_type(x, list[int])
    "#,
);

testcase!(
    test_generic_alias_union_implicit,
    r#"
from typing import TypeVar, assert_type
T = TypeVar('T')
X = T | list[T] | None
def f(x: X[int]):
    assert_type(x, int | list[int] | None)
    "#,
);

testcase!(
    test_generic_alias_union_explicit,
    r#"
from typing import TypeVar, assert_type, TypeAlias
T = TypeVar('T')
X: TypeAlias = T | list[T] | None
def f(x: X[int]):
    assert_type(x, int | list[int] | None)
    "#,
);

testcase!(
    test_type_var_tuple_alias_generic,
    r#"
from typing import Any, assert_type
type X[*T] = tuple[*T]
def f(x: X[int, str]):
    assert_type(x, tuple[int, str])
def f2(x: X):
    assert_type(x, tuple[Any, ...])
    "#,
);

testcase!(
    test_generic_type_var_tuple_alias_implicit,
    r#"
from typing import Any, TypeVarTuple, assert_type
Ts = TypeVarTuple('Ts')
X = tuple[*Ts]
def f(x: X[int, str]):
    assert_type(x, tuple[int, str])
def f2(x: X):
    assert_type(x, tuple[Any, ...])
    "#,
);

testcase!(
    test_generic_type_var_tuple_alias_explicit,
    r#"
from typing import Any, TypeVarTuple, TypeAlias, assert_type
Ts = TypeVarTuple('Ts')
X: TypeAlias = tuple[*Ts]
def f(x: X[int, str]):
    assert_type(x, tuple[int, str])
def f2(x: X):
    assert_type(x, tuple[Any, ...])
    "#,
);

testcase!(
    test_param_spec_alias_generic,
    r#"
from typing import assert_type, Callable
type X[**P] = Callable[P, None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, str], None])
def f2(x: X[[int, str]]):
    assert_type(x, Callable[[int, str], None])
    "#,
);

testcase!(
    test_generic_param_spec_alias_implicit,
    r#"
from typing import ParamSpec, Callable, assert_type
P = ParamSpec('P')
X = Callable[P, None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, str], None])
def f2(x: X[[int, str]]):
    assert_type(x, Callable[[int, str], None])
    "#,
);

testcase!(
    test_generic_param_spec_alias_explicit,
    r#"
from typing import Callable, ParamSpec, TypeAlias, assert_type
P = ParamSpec('P')
X: TypeAlias = Callable[P, None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, str], None])
def f2(x: X[[int, str]]):
    assert_type(x, Callable[[int, str], None])
    "#,
);

testcase!(
    test_param_spec_alias_concatenate_generic,
    r#"
from typing import assert_type, Callable, Concatenate
type X[**P] = Callable[Concatenate[int, P], None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, int, str], None])
    "#,
);

testcase!(
    test_generic_param_spec_alias_concatenate_implicit,
    r#"
from typing import ParamSpec, Callable, Concatenate, assert_type
P = ParamSpec('P')
X = Callable[Concatenate[int, P], None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, int, str], None])
    "#,
);

testcase!(
    test_generic_param_spec_alias_concatenate_explicit,
    r#"
from typing import Callable, ParamSpec, TypeAlias, assert_type, Concatenate
P = ParamSpec('P')
X: TypeAlias = Callable[Concatenate[int, P], None]
def f(x: X[int, str]):
    assert_type(x, Callable[[int, int, str], None])
    "#,
);

testcase!(
    test_generic_alias_callable,
    r#"
from typing import Callable, TypeVar, assert_type
T = TypeVar('T')
X1 = Callable[..., T]
X2 = Callable[[T], str]
def f(x1: X1[int], x2: X2[int]):
    assert_type(x1, Callable[..., int])
    assert_type(x2, Callable[[int], str])
    "#,
);

testcase!(
    test_generic_alias_checked,
    r#"
from typing import Any, assert_type
type X[T1, T2] = dict[T1, T2]
def f(x: X[int]):  # E: Expected 2 type arguments for `X`, got 1
    assert_type(x, dict[int, Any])
    "#,
);

testcase!(
    test_generic_alias_annotated,
    r#"
from typing import Annotated, TypeVar, assert_type
T = TypeVar('T')
X = Annotated[T, 'the world is quiet here']
def f(x: X[int]):
    assert_type(x, int)
    "#,
);

testcase!(
    test_bad_annotated_alias,
    r#"
from typing import TypeAlias
X: TypeAlias = 1  # E: number literal cannot be used in annotations
    "#,
);

testcase!(
    test_not_a_type_alias,
    r#"
TypeAlias = int
x: TypeAlias = 1
    "#,
);

testcase!(
    test_type_alias_import_direct,
    r#"
from typing import TypeAlias
class C: pass
x: TypeAlias = "C"
y: x = C()
"#,
);

testcase!(
    test_type_alias_import_star,
    r#"
from typing import *
class C: pass
x: TypeAlias = "C"
y: x = C()
"#,
);

testcase!(
    test_type_alias_import_module,
    r#"
import typing
class C: pass
x: typing.TypeAlias = "C"
y: x = C()
"#,
);

testcase!(
    test_type_alias_import_named,
    r#"
import typing as tt
class C: pass
x: tt.TypeAlias = "C"
y: x = C()
"#,
);

testcase!(
    test_attribute_access,
    r#"
from typing import TypeAlias
X1 = int
X2: TypeAlias = int
type X3 = int

X1.__add__  # ok
X2.__add__  # ok
X3.__add__  # E: Object of class `TypeAliasType` has no attribute `__add__`
    "#,
);

testcase!(
    test_forward_ref,
    r#"
from typing import TypeAlias, assert_type

X1 = "A"  # Just a normal alias to a str
X2: TypeAlias = "A"  # Forward ref
type X3 = "A"  # Forward ref

class A:
    pass

def f(x1: X1):  # E: Expected a type form, got instance of `Literal['A']`
    pass

def g(x2: X2, x3: X3):
    assert_type(x2, A)
    assert_type(x3, A)
    "#,
);

testcase!(
    test_recursive_alias_explicit,
    r#"
from typing import TypeAlias, assert_type

Alias: TypeAlias = int | list["Alias"]

x: Alias = 1
y: Alias = [1]
z: Alias = [[1, 2]]
"#,
);

testcase!(
    test_recursive_alias_implicit,
    r#"
X = int | list["X"]
x: X = 1
x: X = [1]
x: X = [[1, 2]]
    "#,
);

testcase!(
    test_container_variance,
    r#"
from typing import Iterable
type X1[T] = Iterable[T]
type X2[T] = list[T]
def f1(x: X1[int]):
    pass
def f2(x: X2[int]):
    pass
def g(x: list[bool]):
    f1(x)
    f2(x)  # E: Argument `list[bool]` is not assignable to parameter `x` with type `list[int]`
    "#,
);

testcase!(
    test_type_alias_inside_class,
    r#"
from typing import assert_type
class C:
    type X = int
def f(x: C.X):
    assert_type(x, int)
    "#,
);

// Note: this test documents how Pyre currently treats type aliases inside classes
// when the try to "close" over class type parameters. It is not 100% clear what
// the behavior should be. But we disagree with Pyright, which disallows the use of
// an instance attribute `c.X` and *does* allow both `bad1` (as `list[Any]`) and
// `bad2` (as `list[int]`).
//
// TODO(stroxler) We probably should at least allow the uses that Pyright allows.
testcase!(
    test_parameterized_type_alias_inside_class,
    r#"
from typing import assert_type
class C[T]:
    type X = list[T]
    x: X = []
def f(c: C[int]):
    assert_type(c.x, list[int])
    x: c.X = []
    assert_type(x, list[int])
bad1: C.X  # E: Generic attribute `X` of class `C` is not visible on the class
bad2: C[int].X  # E: Generic attribute `X` of class `C` is not visible on the class
    "#,
);

testcase!(
    test_union_alias,
    r#"
from typing import TypeAlias
StringOrInt: TypeAlias = str | int
x: StringOrInt = 1
"#,
);

fn env_type_alias() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
from typing import TypeAlias
StringOrInt: TypeAlias = str | int
"#,
    )
}

testcase!(
    test_alias_import,
    env_type_alias(),
    r#"
from foo import StringOrInt
x: StringOrInt = 1
"#,
);

testcase!(
    test_union_none,
    r#"
from typing import TypeAlias
NoneOrInt: TypeAlias = None | int
IntOrNone: TypeAlias = int | None
NoneOrStr = None | str
StrOrNone = str | None

a: NoneOrInt = None
b: IntOrNone = 1
c: NoneOrStr = "test"
d: StrOrNone = None
e: NoneOrInt = "test"  # E: `Literal['test']` is not assignable to `int | None`
"#,
);

testcase!(
    test_type_alias_full_name,
    r#"
import typing
from typing import assert_type
X: typing.TypeAlias = int
def f(x: X | str):
    assert_type(x, int | str)
    "#,
);

testcase!(
    test_type_alias_union,
    r#"
from typing import Union, Optional
class Y: pass
X = Union[int, "Y"]
Z = Optional["Y"]
"#,
);

testcase!(
    bug = "The recursive instance of X is resolved to Unknown",
    test_type_alias_recursive,
    r#"
type X = int | list["X"]
x1: X = 1
x2: X = [1]
x3: X = [[1, 2]]
x4: X = [1, [2, 3]]
x5: X = ["foo"]  # Not OK
"#,
);

testcase!(
    bug = "Doesn't detect as a TypeAlias, but it is one. Maybe this is reasonable.",
    test_type_alias_with_string,
    r#"
class Y: pass
class C[T]: pass
X = C["Y"] # E: Expected a type form, got instance of `Literal['Y']`
"#,
);

testcase!(
    test_type_alias_validation,
    r#"
from typing import *
Ts = TypeVarTuple('Ts')
P = ParamSpec('P')
t1: TypeAlias = Unpack[TypedDict]  # E: `Unpack` is not allowed in this context # E: `TypedDict` is not allowed in this context
t2: TypeAlias = P  # E: `ParamSpec` is not allowed in this context
t3: TypeAlias = Unpack[Ts]  # E: `Unpack` is not allowed in this context
t4: TypeAlias = Literal  # E: Expected a type argument for `Literal`
t5: TypeAlias = Ts  # E: `TypeVarTuple` must be unpacked
t6: TypeAlias = Generic  # E: Expected a type argument for `Generic`
t7: TypeAlias = Protocol  # E: Expected a type argument for `Protocol`
t8: TypeAlias = Generic[int]  # E: `Generic` is not allowed in this context
t9: TypeAlias = Protocol[int]  # E: `Protocol` is not allowed in this context
t10: TypeAlias = Final  # E: Expected a type argument for `Final`
t11: TypeAlias = Final[int]  # E: `Final` is not allowed in this context
t12: TypeAlias = TypeAlias  # OK
t13: TypeAlias = [int][0]  # E: invalid subscript expression cannot be used in annotations
"#,
);

testcase!(
    test_type_alias_argparse,
    r#"
from typing import Callable, Any
def foo(x: Callable[[str], Any]) -> None:
    pass

foo(str)
    "#,
);

testcase!(
    test_type_alias_generics,
    r#"
from typing import Generic, Hashable, Iterable, TypeVar, TypeAlias

_Node = TypeVar("_Node", bound=Hashable)
_NBunch: TypeAlias = _Node | Iterable[_Node] | None

class DiDegreeView(Generic[_Node]):
    def __init__(
        self,
        nbunch: _NBunch[_Node] = None,
    ) -> None: ...
    "#,
);

testcase!(
    test_type_alias_bin_op,
    r#"
from typing import TypeAlias

class B: ...
R: TypeAlias = B
class C: ...
class F:
    T: TypeAlias = C
    def memmove(self, arg: T | R) -> None: ...
    "#,
);

testcase!(
    test_type_alias_implicit_bad_syntax,
    r#"
def get_class() -> type[object]:
    return object

get_class()
foo = get_class() # Cannot be an alias because of syntax
bar: type[object] = get_class()
"#,
);

testcase!(
    test_union_with_implicit_generic_alias,
    r#"
from typing import Any, TypeVar, assert_type
T = TypeVar('T')
X = list[T]
def f(x1: X | None, x2: X[int] | None):
    assert_type(x1, list[Any] | None)
    assert_type(x2, list[int] | None)
    "#,
);

testcase!(
    test_union_with_implicit_generic_alias_union,
    r#"
from typing import Any, TypeVar, assert_type
T = TypeVar('T')
Y = list[T] | set[T]
def f(y1: Y | None, y2: Y[int] | None):
    assert_type(y1, list[Any] | set[Any] | None)
    assert_type(y2, list[int] | set[int] | None)
    "#,
);

testcase!(
    test_implicit_alias_of_typevar_type,
    r#"
from typing import TypeVar
T = TypeVar('T', bound=int)
X = type[T]
def f(x: X[bool]) -> bool:
    return x()
def g(x: type[T][int]):  # E: invalid subscript expression
    pass
def h(x: X[str]):  # E: `str` is not assignable to upper bound `int`
    pass
    "#,
);

testcase!(
    test_bad_type_variable_aliases,
    r#"
from typing import ParamSpec, TypeVarTuple, Unpack
P = ParamSpec('P')
Ts = TypeVarTuple('Ts')
Error1 = type[P]  # E: `ParamSpec` is not allowed
Error2 = type[Ts]  # E: `TypeVarTuple` must be unpacked
Error3 = type[Unpack[Ts]]  # E: `Unpack` is not allowed
    "#,
);

testcase!(
    test_bound_mismatch,
    r#"
from typing import TypeAlias, TypeVar

T = TypeVar('T', bound=int)

ImplicitAlias = list[T]
ExplicitAlias: TypeAlias = list[T]
type ScopedAlias[S: int] = list[S]

def f(
    x: ImplicitAlias[str],  # E: `str` is not assignable to upper bound `int` of type variable `T`
    y: ExplicitAlias[str],  # E: `str` is not assignable to upper bound `int`
    z: ScopedAlias[str],  # E: `str` is not assignable to upper bound `int`
): ...
    "#,
);

testcase!(
    test_constraint_mismatch,
    r#"
from typing import TypeAlias, TypeVar

T = TypeVar('T', int, bytes)

ImplicitAlias = list[T]
ExplicitAlias: TypeAlias = list[T]
type ScopedAlias[S: (int, bytes)] = list[S]

def f(
    x: ImplicitAlias[str],  # E: `str` is not assignable to upper bound `bytes | int` of type variable `T`
    y: ExplicitAlias[str],  # E: `str` is not assignable to upper bound `bytes | int`
    z: ScopedAlias[str],  # E: `str` is not assignable to upper bound `bytes | int`
): ...
    "#,
);

testcase!(
    test_specialize_typevar_with_typevar,
    r#"
from typing import TypeAlias, TypeVar

T = TypeVar('T', bound=int)
S = TypeVar('S', bound=bool)
U = TypeVar('U', bound=str)

X: TypeAlias = list[T]
Y: TypeAlias = X[S]
Z: TypeAlias = X[U]  # E: `U` is not assignable to upper bound `int`
    "#,
);

testcase!(
    test_invalid_scoped_alias,
    r#"
def test():
    type X = int  # E: `type` statement is not allowed in this context
    "#,
);

testcase!(
    test_type_alias_base_class,
    r#"
from typing import TypeAlias

# Test that scoped type aliases cannot be used as base classes
type X = int
Y: TypeAlias = int
Z = int

class C1(X): pass  # E: Cannot use scoped type alias `X` as a base class
class C2(Y): pass  # Should work - legacy explicit type alias
class C3(Z): pass  # Should work - legacy implicit type alias
    "#,
);

testcase!(
    test_string_annotations,
    r#"
from typing import Annotated, Callable, Dict, List, Tuple, Type
from collections.abc import Callable as Callable2
X1 = Annotated["A", "foo"]
X2 = Callable[..., "A"]
X3 = Callable2[..., "A"]
X4 = Dict[str, "A"]
X5 = dict[str, "A"]
X6 = List["A"]
X7 = list["A"]
X8 = Tuple["A", ...]
X9 = tuple["A", ...]
XA = type["A"]
XB = Type["A"]
class A:
    pass
    "#,
);

testcase!(
    test_generic_typealias_of_typealiastype,
    r#"
from typing import TypeAlias, TypeAliasType, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2", bound=str | bytes)

Spam1 = TypeAliasType("Spam1", T2 | type[T1], type_params=(T1, T2))
Spam2: TypeAlias = Spam1[T1, T2]

x1: Spam1[int, str] = int
x2: Spam2[int, str] = int
    "#,
);

testcase!(
    test_generic_typealias_of_scopedtypealias,
    r#"
from typing import TypeAlias, TypeAliasType, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2", bound=str | bytes)

type Spam1[T1, T2] = T2 | type[T1]
Spam2: TypeAlias = Spam1[T1, T2]

x1: Spam1[int, str] = int
x2: Spam2[int, str] = int
    "#,
);

testcase!(
    test_generic_typealias_of_explicit_typealias,
    r#"
from typing import TypeAlias, TypeAliasType, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2", bound=str | bytes)

Spam1: TypeAlias = type[T1] | T2
Spam2: TypeAlias = Spam1[T1, T2]

x1: Spam1[int, str] = int
x2: Spam2[int, str] = int
    "#,
);

testcase!(
    test_variable_in_function_is_not_type_alias,
    r#"
from typing import TypeVar, Generic
class C:
    @classmethod
    def make(cls) -> C:
        raise NotImplementedError()
T = TypeVar("T", bound=C)
class UseC(Generic[T]):
    _class: type[T] | None
    def use(self) -> None:
        current = self._class # this is not a type alias
        if current is not None:
            current.make()
    "#,
);

fn env_with_alias() -> TestEnv {
    TestEnv::one(
        "foo",
        r#"
type TA = int | str

def f(x: TA) -> TA:
  return x
"#,
    )
}

testcase!(
    test_alias_union_name,
    env_with_alias(),
    r#"
from foo import TA, f
from typing import Callable

val1: int | str = 1
val2: TA = 1

# Union names are only shown when nested in another type

f(object())  # E: Argument `object` is not assignable to parameter `x` with type `int | str` in function `foo.f`
f(val1)
f(val2)
x1: TA = object()  # E: `object` is not assignable to `int | str`
x2: TA = val1
x3: TA = val2
c1: Callable[[int], int] = f  # E: `(x: TA) -> TA` is not assignable to `(int) -> int`

# Union names are lost when flattened into another union

class C: pass
def f2(x: TA | C) -> TA | C:
  return x

f2(object())  # E: Argument `object` is not assignable to parameter `x` with type `C | int | str` in function `f2`
f2(val1)
f2(val2)
f2(C())
x4: TA | C = object()  # E: `object` is not assignable to `C | int | str`
x5: TA | C = val1
x6: TA | C = val2
x7: TA | C = C()
c2: Callable[[int], int] = f2  # E: `(x: C | int | str) -> C | int | str` is not assignable to `(int) -> int`
    "#,
);
