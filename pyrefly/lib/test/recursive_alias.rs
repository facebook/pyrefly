/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_basic,
    r#"
from typing import Union
X = Union[int, list["X"]]

x1: X = 1
x2: X = [1]
x3: X = [[1]]

x4: X = ["oops"]  # E: not assignable
    "#,
);

testcase!(
    test_display,
    r#"
from typing import reveal_type, TypeAlias, Union

X1 = Union[int, list["X1"]]
X2: TypeAlias = Union[int, list["X2"]]
type X3 = int | list["X3"]

Y1 = Union[int, list["Y2"]]
Y2 = Union[int, list["Y1"]]

def f(x1: X1, x2: X2, x3: X3, y1: Y1, y2: Y2):
    reveal_type(x1)  # E: int | list[X1]
    reveal_type(x2)  # E: int | list[X2]
    reveal_type(x3)  # E: int | list[X3]
    reveal_type(y1)  # E: int | list[Y2]
    reveal_type(y2)  # E: int | list[Y1]
    "#,
);

testcase!(
    test_iterate,
    r#"
type X = int | list[X]
def f(x: X) -> X | None:
    if isinstance(x, list):
        for y in x:
            if y:
                return y
    "#,
);

testcase!(
    test_import,
    TestEnv::one("foo", "type X = int | list[X]"),
    r#"
import foo
x1: foo.X = [[1]]
x2: foo.X = [["oops"]]  # E: not assignable
    "#,
);

testcase!(
    test_from_import,
    TestEnv::one("foo", "type X = int | list[X]"),
    r#"
from foo import X
x1: X = [[1]]
x2: X = [["oops"]]  # E: not assignable
    "#,
);

testcase!(
    test_equivalent,
    r#"
type X = int | list[X]
type Y = int | list[Y]
def f(x: X) -> Y:
    return x
    "#,
);

testcase!(
    test_class_attr,
    r#"
class C:
    type X = int | list[C.X]
x1: C.X = [[1]]
x2: C.X = [["oops"]]  # E: not assignable
    "#,
);

testcase!(
    bug = "Fails to resolve forward ref",
    test_unqualified_class_attr_ref,
    r#"
class C:
    type X = int | list[X]  # E: Could not find name `X`
    "#,
);

testcase!(
    test_generic_scoped,
    r#"
from typing import reveal_type

type X[T] = T | list[X[T]]

x1: X[int] = [[1]]
x2: X[str] = [[1]]  # E: not assignable

def f[T](x: X[T]):
    reveal_type(x)  # E: list[X[T]] | T
    "#,
);

testcase!(
    test_generic_legacy,
    r#"
from typing import reveal_type, TypeVar, Union

T = TypeVar("T")

X = Union[T, list[X[T]]]

x1: X[int] = [[1]]
x2: X[str] = [[1]]  # E: not assignable

def f[T](x: X[T]):
    reveal_type(x)  # E: list[X[T]] | T
    "#,
);

testcase!(
    test_generic_typealiastype,
    r#"
from typing import reveal_type, TypeAliasType, TypeVar, Union

T = TypeVar("T")

X = TypeAliasType("X", T | list[X[T]], type_params=(T,))

x1: X[int] = [[1]]
x2: X[str] = [[1]]  # E: not assignable

def f[T](x: X[T]):
    reveal_type(x)  # E: list[X[T]] | T
    "#,
);

testcase!(
    test_illegal_subscript,
    r#"
from typing import Union
type X = int | list[X[int]]  # E: `type[X]` is not subscriptable
Y = Union[int, list[Y[int]]]  # E: `type[Y]` is not subscriptable
    "#,
);

testcase!(
    test_subscript_twice,
    r#"
type X[T] = int | list[X[int][int]]  # E: `type[X[int]]` is not subscriptable
    "#,
);

testcase!(
    test_bad_targ,
    r#"
type X[T] = int | list[X[0]]  # E: got instance of `Literal[0]`
    "#,
);

testcase!(
    test_violate_bound,
    r#"
type X[T: int] = int | list[X[str]]  # E: `str` is not assignable to upper bound `int` of type variable `T`
    "#,
);

testcase!(
    test_generic_multiple_tparams,
    r#"
from typing import reveal_type

type X[K, V] = dict[K, V] | list[X[str, V]]

x1: X = {0: 1}
x2: X[int, int] = {0: 1}
x3: X[str, int] = {0: 1}  # E: `int` is not assignable to `str`

x4: X = [{'ok': 1}]
x5: X[int, int] = [{'ok': 1}]
x6: X = [{0: 1}]  # E: not assignable
x7: X[int, int] = [{'no': 3.14}]  # E: not assignable

def f[K, V](x1: X[K, V], x2: X[int, int]):
    reveal_type(x1)  # E: dict[K, V] | list[X[str, V]]
    reveal_type(x2)  # E: dict[int, int] | list[X[str, int]]
    "#,
);

testcase!(
    test_nongeneric_subscriptable,
    r#"
from typing import reveal_type
type X = list[list[int]] | list[X]
def f(x: X):
    for y in x:
        reveal_type(y[0])  # E: int | list[int] | X
    "#,
);

testcase!(
    test_promote_implicit_any,
    r#"
type X[T] = int | list[X]  # unparameterized `X` reference is implicitly `X[Any]`
def f(x: X[str]) -> X[int]:
    return [x]
    "#,
);

testcase!(
    test_error_implicit_any,
    TestEnv::new().enable_implicit_any_error(),
    r#"
type X[T] = int | list[X]  # E: Cannot determine the type parameter `T` for generic type alias `X`
def f(x: X[str]) -> X[int]:
    return [x]
    "#,
);

testcase!(
    bug = "We should report the bound violation in `C[R]`",
    test_check_class_tparam_bound,
    r#"
class A: pass
class C[T: A]: pass
type R = int | C[R]
    "#,
);

testcase!(
    test_cyclic,
    r#"
type W = W  # E: cyclic self-reference in `W`
type X = int | X  # E: cyclic self-reference in `X`
type Y = int | Z
type Z = int | Y  # E: cyclic self-reference in `Z`
    "#,
);

testcase!(
    test_recursive_function_type,
    r#"
from typing import Callable
type F = Callable[[int], None | F]
def g(f: F): pass
def h1(x: int) -> None:
    pass
def h2(x: int) -> Callable[[int], None]:
    return h1
def h3(x: int) -> int:
    return x
g(h1)
g(h2)
g(h3)  # E: not assignable
    "#,
);
