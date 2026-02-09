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
    bug = "Fails to catch type error in x2, bad reveal_type",
    test_generic_scoped,
    r#"
from typing import reveal_type

type X[T] = T | list[X[T]]

x1: X[int] = [[1]]
x2: X[str] = [[1]]

def f[T](x: X[T]):
    reveal_type(x)  # E: list[Unknown] | T
    "#,
);

testcase!(
    bug = "Fails to catch type error in x2, bad reveal_type",
    test_generic_legacy,
    r#"
from typing import reveal_type, TypeVar, Union

T = TypeVar("T")

X = Union[T, list[X[T]]]

x1: X[int] = [[1]]
x2: X[str] = [[1]]

def f[T](x: X[T]):
    reveal_type(x)  # E: list[Unknown] | T
    "#,
);

testcase!(
    bug = "Fails to catch illegal subscription",
    test_illegal_subscript,
    r#"
from typing import Union
type X = int | list[X[int]]
Y = Union[int, list[Y[int]]]
    "#,
);

testcase!(
    bug = "Fails to catch errors, bad reveal_type",
    test_generic_multiple_tparams,
    r#"
from typing import reveal_type

type X[K, V] = dict[K, V] | list[X[str, V]]

x1: X = {0: 1}
x2: X[int, int] = {0: 1}
x3: X[str, int] = {0: 1}  # E: `dict[int, int]` is not assignable to `dict[str, int] | list[Unknown]`

x4: X = [{'ok': 1}]
x5: X[int, int] = [{'ok': 1}]
x6: X = [{0: 1}]  # should error!
x7: X[int, int] = [{'no': 3.14}]  # should error!

def f[K, V](x1: X[K, V], x2: X[int, int]):
    reveal_type(x1)  # E: dict[K, V] | list[Unknown]
    reveal_type(x2)  # E: dict[int, int] | list[Unknown]
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
