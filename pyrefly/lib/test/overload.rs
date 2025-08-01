/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_py,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

def f(x):
    return x

assert_type(f(1), int)

def anywhere():
    assert_type(f(1), int)
    "#,
);

testcase!(
    test_branches,
    r#"
from typing import assert_type, overload
def test(x: bool):
    if x:
        def f(x: str) -> bytes: ...
    else:
        @overload
        def f(x: int) -> int: ...
        @overload
        def f(x: str) -> str: ...
        def f(x: int | str) -> int | str:
            return x
    def g(x: str):
        assert_type(f(x), bytes | str)
    "#,
);

fn env_with_stub() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "foo",
        "foo.pyi",
        r#"
from typing import overload

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...
    "#,
    );
    t
}

testcase!(
    test_pyi,
    env_with_stub(),
    r#"
from typing import assert_type
import foo
assert_type(foo.f(1), int)
    "#,
);

testcase!(
    test_protocol,
    r#"
from typing import Protocol, assert_type, overload

class P(Protocol):
    @overload
    def m(self, x: int) -> int: ...
    @overload
    def m(self, x: str) -> str: ...

def test(o: P):
    assert_type(o.m(1), int)
    "#,
);

testcase!(
    test_method,
    r#"
from typing import assert_type, overload

class C:
    @overload
    def m(self, x: int) -> int: ...
    @overload
    def m(self, x: str) -> str: ...
    def m(self, x: int | str) -> int | str:
        return x

def test(o: C):
    assert_type(o.m(1), int)
    "#,
);

testcase!(
    test_overload_arg_errors,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str: ...

def g(x: str) -> int: ...
def h(x: str) -> str: ...

assert_type(f(g(0)), int) # E: Argument `Literal[0]` is not assignable to parameter `x` with type `str`
assert_type(f(h(0)), str) # E: Argument `Literal[0]` is not assignable to parameter `x` with type `str`
"#,
);

testcase!(
    test_overload_missing_implementation,
    r#"
from typing import overload, assert_type

@overload
def f(x: int) -> int: ... # E: Overloaded function must have an implementation
@overload
def f(x: str) -> str: ...

# still behaves like an overload
assert_type(f(0), int)
assert_type(f(""), str)
"#,
);

testcase!(
    test_overload_static_config,
    r#"
from typing import overload, assert_type
import sys

@overload
def f(x: int) -> int: ... # E: Overloaded function must have an implementation

if sys.version_info >= (3, 11):
    @overload
    def f(x: str) -> str: ...
else:
    @overload
    def f(x: int, int) -> bool: ...

if sys.version_info >= (3, 12):
    @overload
    def f() -> None: ...

assert_type(f(0), int)
assert_type(f(""), str)
assert_type(f(), None)
f(0, 0) # E: No matching overload found
"#,
);

testcase!(
    test_only_one_overload,
    r#"
from typing import overload, Protocol

@overload
def f(x: int) -> int: ...  # E: Overloaded function needs at least two @overload declarations
def f(x: int) -> int:
    return x

@overload
def g(x: int) -> int: ...  # E: Overloaded function must have an implementation  # E: Overloaded function needs at least two @overload declarations

class P(Protocol):
    @overload
    def m(x: int) -> int: ...  # E: Overloaded function needs at least two @overload declarations
"#,
);

testcase!(
    test_overload_ignore,
    r#"
from typing import Never, overload, assert_type

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

x = f("foo") # type: ignore
# intentionally blank: make sure we don't ignore the assert_type below
assert_type(x, str)
"#,
);

testcase!(
    test_typeguard,
    r#"
from typing import assert_type, overload, TypeGuard

class Animal: ...
class Mammal(Animal): ...
class Cat(Mammal): ...
class Bird(Animal): ...
class Robin(Bird): ...

@overload
def f(x: Mammal) -> TypeGuard[Cat]: ...
@overload
def f(x: Bird) -> TypeGuard[Robin]: ...
def f(x: Animal) -> bool: ...

class A:
    @overload
    def f(self, x: Mammal) -> TypeGuard[Cat]: ...
    @overload
    def f(self, x: Bird) -> TypeGuard[Robin]: ...
    def f(self, x: Animal) -> bool: ...

def g(meow: Mammal, chirp: Bird):
    if f(meow):
        assert_type(meow, Cat)
    if A().f(chirp):
        assert_type(chirp, Robin)
    "#,
);

testcase!(
    test_classmethod,
    r#"
from typing import assert_type, overload
class A:
    @overload
    @classmethod
    def f(cls, x: int) -> int: ...

    @overload
    @classmethod
    def f(cls, x: str) -> str: ...

    @classmethod
    def f(cls, x: int | str) -> int | str:
        return x

assert_type(A().f(1), int)
    "#,
);

testcase!(
    test_invalid_decoration,
    r#"
from typing import overload

def decorate(f) -> float:
    return 0

@overload
def f(x: str) -> str: ...

@decorate
@overload
def f(x: int) -> int: ...  # E: `f` has type `float` after decorator application, which is not callable

def f(x: str | int) -> str | int:
    return x
    "#,
);

testcase!(
    test_decoration,
    r#"
from typing import Callable, assert_type, overload

def decorate(f) -> Callable[[int], int]:
    return lambda x: x

@overload
@decorate
def f(x: bytes) -> bytes: ...

@overload
def f(x: str) -> str: ...

def f(x: object) -> object:
    return x

assert_type(f(0), int)
f(b"")  # E: No matching overload found for function `f`
    "#,
);

testcase!(
    test_final_decoration,
    r#"
from typing import assert_type, final, overload

@overload
@final
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

def f(x: int | str) -> int | str:
    return x

assert_type(f(0), int)
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_1,
    r#"
from collections import defaultdict
from typing import DefaultDict
x: DefaultDict[int, list[int]] = defaultdict(list)
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_2,
    r#"
from collections import defaultdict
x: dict[int, int] = defaultdict(int)
    "#,
);

testcase!(
    test_defaultdict_constructor_overload_3,
    r#"
import collections
std_aggs: dict[int, tuple[list[str], list[str]]] = collections.defaultdict(
    lambda: ([], [])
)
std_aggs[0][1].append('foo')
    "#,
);

testcase!(
    test_constructor_overload_with_hint,
    r#"
from typing import Callable, overload
class defaulty[K, V]:
    @overload
    def __init__(self: defaulty[str, V], **kwargs: V) -> None: ...
    @overload
    def __init__(self, default_factory: Callable[[], V] | None, /) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        return None
badge: defaulty[bool, list[str]] = defaulty(list)
    "#,
);

testcase!(
    test_pass_generic_class_to_overload,
    r#"
from typing import Iterable, Literal, overload, Self
from _typeshed import SupportsAdd

@overload
def f(x: Iterable[Literal[1]]) -> None: ...
@overload
def f(x: Iterable[SupportsAdd]) -> None: ...
def f(x) -> None: ...

class C[T](Iterable[T]):
    def __new__(cls, x: T) -> Self: ...

def g(x: int):
    f(C(x))
    "#,
);

testcase!(
    test_overload_exponential,
    r#"
# This used to take an exponential amount of time to type check

from typing import overload, Any
class X: ...

@overload
def f(a: int) -> X: ...
@overload
def f(a: str) -> X: ...
def f(a: Any) -> X: return X()

def exponential() -> Any:
    f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(X())))))))))))))))))))))))) # E: # E: # E: # E: # E: # E: # E: # E: # E: # E: # E: # E:
"#,
);

testcase!(
    test_implementation_with_overload,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...

@overload
def f(x: int | str) -> int | str: # E: @overload decorator should not be used on function implementation
    return x
    "#,
);

testcase!(
    test_implementation_before_overload,
    r#"
from typing import overload

def f(x: int | str) -> int | str: # E: @overload declarations must come before function implementation
    return x

@overload
def f(x: int) -> int: ...

@overload
def f(x: str) -> str: ...
    "#,
);

testcase!(
    test_overload_with_docstring,
    r#"
from typing import overload, Any

@overload
def foo(a: int) -> int: ...  
@overload
def foo(a: str) -> str:
    """Docstring"""
def foo(*args, **kwargs) -> Any:
    pass
    "#,
);

testcase!(
    test_overload_with_docstring2,
    r#"
from typing import overload, Any

@overload
def foo(a: int) -> int: ...  
@overload
def foo(a: str) -> str: 
    """Docstring"""
    return 123             # E: Returned type `Literal[123]` is not assignable to declared return type `str`
def foo(*args, **kwargs) -> Any:
    pass
    "#,
);

testcase!(
    test_overload_with_docstring3,
    r#"
def foo() -> int: # E: Function declared to return `int` but is missing an explicit `return`
    """hello"""
    "#,
);

testcase!(
    test_return_consistency,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...  # E: Overload return type `str` is not assignable to implementation return type `int`
def f(x: int | str) -> int:
    return int(x)
    "#,
);

testcase!(
    test_generic_overloads_are_consistent,
    r#"
from typing import Any, overload

@overload
def f(a: int, b: Any) -> float: ...
@overload
def f[T](a: float, b: T) -> T: ...
def f[T](a: float, b: T) -> T: ...
    "#,
);

testcase!(
    test_overloads_are_consistent_with_generic_impl,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f[T](x: T) -> T:
    return x
    "#,
);

testcase!(
    test_param_consistency,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> int: ...  # E: Implementation signature `(x: int) -> int` does not accept all arguments that overload signature `(x: str) -> int` accepts
def f(x: int) -> int:
    return x
    "#,
);

testcase!(
    test_generic_implementation_multiple_typevars,
    r#"
from typing import overload

@overload
def f(x: int, y: str) -> tuple[str, int]: ...
@overload
def f(x: bool, y: float) -> tuple[float, bool]: ...
def f[T1, T2](x: T1, y: T2) -> tuple[T2, T1]:
    return (y, x)

@overload
def g(x: int, y: str) -> tuple[str, int]: ...  # E: `tuple[str, int]` is not assignable to implementation return type `tuple[int, str]`
@overload
def g(x: bool, y: float) -> tuple[float, bool]: ...  # E: `tuple[float, bool]` is not assignable to implementation return type `tuple[bool, float]`
def g[T1, T2](x: T1, y: T2) -> tuple[T1, T2]:
    return (x, y)
    "#,
);

testcase!(
    test_generic_overload_nongeneric_impl,
    r#"
from typing import Any, overload

@overload
def f[T](x: T, y=None) -> T: ...  # E: does not accept all arguments  # E: not assignable to implementation return type
@overload
def f(x, y) -> Any: ...
def f(x: int, y=None) -> int:
    return x

@overload
def g[T: int](x: T, y=None) -> T: ...
@overload
def g(x, y) -> Any: ...
def g(x: int, y=None) -> int:
    return x
    "#,
);
