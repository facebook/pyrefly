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
    test_final_decoration_on_top_level_function,
    r#"
from typing import assert_type, final, overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
@final  # E: `@final` can only be used on methods
def f(x: int | str) -> int | str:
    return x

assert_type(f(0), int)
    "#,
);

testcase!(
    test_overload_inconsistent_override_final,
    r#"
from typing import overload, override, Any, final

class Base:
    def f(x: Any) -> Any: pass
    def f2(x: Any) -> Any: pass

class C(Base):
    # OK
    @override
    @overload
    def f(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @overload
    def f(x: str) -> str: ...

    # not on the first overload
    @overload
    def f2(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @overload
    @override
    def f2(x: str) -> str: ...  # E: `@override` should be applied to the first overload only.

    # OK
    @overload
    def f3(x: int) -> int: ...
    @overload
    def f3(x: str) -> str: ...
    @final
    def f3(x: int | str):
        return x

    # not on the implementation
    @final
    @overload
    def f3(x: int) -> int: ...  # E: `@final` should only be applied to the implementation of an overloaded function
    @overload
    def f3(x: str) -> str: ...
    def f3(x: int | str):
        return x
    "#,
);

testcase!(
    test_overload_inconsistent_staticmethod_classmethod,
    r#"
from typing import overload

class C:
    # missing from an overload
    @overload
    def f(x: int) -> int: ...  # E: Overloaded function must have an implementation  # E: If `@staticmethod` is present on one overload, all overloads must have that decorator
    @staticmethod
    @overload
    def f(x: str) -> str: ...

    # OK
    @staticmethod
    @overload
    def f2(x: int) -> int: ...  # E: Overloaded function must have an implementation
    @staticmethod
    @overload
    def f2(x: str) -> str: ...

    # OK
    @classmethod
    @overload
    def f3(x: int) -> int: ...
    @classmethod
    @overload
    def f3(x: str) -> str: ...
    @classmethod
    def f3(x: int | str):
        return x

    # missing from implementation
    @classmethod
    @overload
    def f4(x: int) -> int: ...
    @classmethod
    @overload
    def f4(x: str) -> str: ...
    def f4(x: int | str):  # E: If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation
        return x

    # missing from an overload
    @classmethod
    @overload
    def f5(x: int) -> int: ...
    @overload
    def f5(x: str) -> str: ...  # E: If `@classmethod` is present on any overload or the implementation, it should be on every overload and the implementation
    @classmethod
    def f5(x: int | str):
        return x
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
from typing import Iterable, Iterator, Literal, overload, Self
from _typeshed import SupportsAdd

@overload
def f(x: Iterable[Literal[1]]) -> None: ...
@overload
def f(x: Iterable[SupportsAdd]) -> None: ...
def f(x) -> None: ...

class C[T](Iterable[T]):
    def __new__(cls, x: T) -> Self: ...
    def __iter__(self) -> Iterator[T]: ...

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

testcase!(
    test_overload_typeddict_errors,
    r#"
from typing import Any, TypedDict, overload, assert_type

class TD(TypedDict):
    x: int

@overload
def foo(d: TD) -> None: ...
@overload
def foo(d: int) -> None: ...
def foo(d: TD | int) -> None: ...

assert_type(foo({ "x": "foo" }), Any) # E: No matching overload found for function `foo`
    "#,
);

testcase!(
    test_generic_impl_input_inconsistent,
    r#"
from typing import overload, TypeVar
T = TypeVar("T")
S = TypeVar("S")

# The implementation signature is intentionally inconsistent with this overload signature
# (`exception` is kw-only in the implementation) so that we can test how legacy TypeVars are
# printed in the error message.
@overload
def catch(exception: T) -> T: ...  # E: Implementation signature `(f: S | None = None, *, exception: T) -> S | T` does not accept all arguments that overload signature `(exception: T) -> T` accepts

@overload
def catch(f: S, *, exception: T) -> S | T: ...

def catch(f: S | None = None, *, exception: T) -> S | T: ...
    "#,
);

testcase!(
    test_abstract,
    r#"
from abc import abstractmethod
from typing import Literal, overload

class Derp:
    @overload
    @abstractmethod
    def f(self, m: Literal["x"] = "x") -> int: ...
    @overload
    @abstractmethod
    def f(self, m: str) -> str: ...
    @abstractmethod
    def f(self, m: str = "x") -> int | str: ...

def test(x: Derp, m: Literal["y"] = "y") -> str:
    return x.f(m)
    "#,
);

testcase!(
    test_expand_union,
    r#"
from typing import assert_type, overload
@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

def g(x: int | str):
    y = f(x)
    assert_type(y, int | str)
    "#,
);

testcase!(
    test_expand_second_arg,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: int, y: str) -> str: ...
def f(x: int, y: int | str) -> int | str:
    return y

def g(y: int | str):
    assert_type(f(0, y), int | str)
    "#,
);

testcase!(
    test_expand_twice,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: int, y: str) -> int: ...
@overload
def f(x: str, y: int) -> str: ...
@overload
def f(x: str, y: str) -> str: ...
def f(x: int | str, y: int | str) -> int | str:
    return x

def g(x: int | str, y: int | str):
    assert_type(f(x, y), int | str)
    "#,
);

testcase!(
    test_expand_bool,
    r#"
from typing import assert_type, overload, Literal

@overload
def f(x: Literal[True]) -> Literal['True']: ...
@overload
def f(x: Literal[False]) -> Literal['False']: ...
def f(x: bool) -> str:
    return str(x)

def g(x: bool):
    assert_type(f(x), Literal['True', 'False'])
    "#,
);

testcase!(
    test_expand_enum,
    r#"
from enum import Enum
from typing import assert_type, overload, Literal

class E(Enum):
    X = 1
    Y = 2

@overload
def f(x: Literal[E.X]) -> Literal['X']: ...
@overload
def f(x: Literal[E.Y]) -> Literal['Y']: ...
def f(x: E) -> str:
    return x.name

def g(x: E):
    assert_type(f(x), Literal['X', 'Y'])
    "#,
);

testcase!(
    test_expand_one_member_enum,
    r#"
from enum import Enum
from typing import assert_type, overload, Literal

class E(Enum):
    X = 1

@overload
def f(x: Literal[E.X]) -> Literal['X']: ...
@overload
def f(x: str) -> str: ...
def f(x: E | str) -> str:
    return str(x)

def g(x: E):
    assert_type(f(x), Literal['X'])
    "#,
);

testcase!(
    test_expand_type_union,
    r#"
from typing import assert_type, overload

class A: ...
class B: ...

@overload
def f(x: type[A]) -> A: ...
@overload
def f(x: type[B]) -> B: ...
def f(x: type[A | B]) -> A | B:
    return x()

def g(x: type[A | B]):
    assert_type(f(x), A | B)
    "#,
);

testcase!(
    test_expand_tuple,
    r#"
from typing import assert_type, overload, Literal

@overload
def f(x: tuple[int, Literal[True]]) -> str: ...
@overload
def f(x: tuple[int, Literal[False]]) -> int: ...
def f(x: tuple[int, bool]) -> int | str:
    return str(x[0]) if x[1] else x[0]

def g(x: tuple[int, bool]):
    assert_type(f(x), str | int)
    "#,
);

testcase!(
    test_wrong_arity,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, y: int) -> int: ...
def f(x: int, y: int = 0) -> int:
    return x + y

f(0, 1, 2)  # E: (x: int, y: int) -> int [closest match]
    "#,
);

testcase!(
    test_unpack_nothing,
    r#"
from typing import assert_type, overload

@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: str) -> str: ...
@overload
def f(x: float) -> float: ...
def f(x, y=0) -> int | str | float: ...

assert_type(f("", *()), str)
assert_type(f(0.0, **{}), float)
    "#,
);

testcase!(
    test_unpack_required,
    r#"
from typing import assert_type, overload

@overload
def f(x: str, /) -> str: ...
@overload
def f(x: int, y: int, /) -> int: ...
def f(x, y=0) -> str | int: ...

def g(*args: int):
    assert_type(f(0, *args), int)
    "#,
);

testcase!(
    test_select_using_args,
    r#"
from typing import assert_type, overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, *args: int) -> str: ...
def f(x: int, *args: int) -> int | str: ...

def g(x: int, y: tuple[()], z: tuple[int, ...]):
    assert_type(f(x, *y), int)
    assert_type(f(x, *z), str)
    "#,
);

testcase!(
    test_select_using_kwargs,
    r#"
from typing import assert_type, overload, TypedDict

@overload
def f(x: int) -> int: ...
@overload
def f(x: int, **kwargs: int) -> str: ...
def f(x: int, **kwargs: int) -> int | str: ...

class Empty(TypedDict):
    pass

def g(x: int, y: Empty, z: dict[str, int]):
    assert_type(f(x, **y), int)
    assert_type(f(x, **z), str)
    "#,
);

testcase!(
    test_materialization_eliminates_overload,
    r#"
from typing import Any, assert_type, overload

@overload
def f(x: list[Any]) -> int: ...
@overload
def f(x: list[str]) -> str: ...
def f(x: list[Any]) -> int | str: ...

def g(x: list[Any]):
    # Because all materializations of `list[Any]` (the argument) are assignable to `list[Any]` (the
    # type of parameter `x` in the first overload), we can eliminate the second overload, leaving
    # us with the first overload with return type `int`.
    assert_type(f(x), int)
    "#,
);

testcase!(
    test_materialization_does_not_eliminate_overload,
    r#"
from typing import Any, assert_type, overload

@overload
def f(x: list[int]) -> int: ...
@overload
def f(x: list[str]) -> str: ...
def f(x: Any) -> int | str: ...

def g(x: list[Any]):
    # There's no overload for which all materializations of `list[Any]` are assignable to the
    # parameter type, so we keep all overloads. Their return types are not equivalent, so we fall
    # back to `Any`.
    assert_type(f(x), Any)

    "#,
);

testcase!(
    test_callable_param_materialization,
    r#"
from typing import Any, assert_type, Callable, Never, overload

@overload
def f1(x: Callable[[int], None]) -> int: ...
@overload
def f1(x: Callable[[str], None]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Callable[[Any], None]) -> int: ...
@overload
def f2(x: Callable[[str], None]) -> str: ...
def f2(x: Any) -> int | str: ...

@overload
def f3(x: Callable[[Never], None]) -> int: ...
@overload
def f3(x: Callable[[str], None]) -> str: ...
def f3(x: Any) -> int | str: ...

def g(x: Callable[[Any], None]):
    assert_type(f1(x), Any)
    assert_type(f2(x), int)
    assert_type(f3(x), int)
    "#,
);

testcase!(
    test_callable_ellipsis_materialization,
    r#"
from typing import Any, assert_type, Callable, overload, Protocol

class EverythingCallback(Protocol):
    def __call__(self, *args, **kwargs) -> None: ...

@overload
def f1(x: EverythingCallback) -> int: ...
@overload
def f1(x: Callable[[], None]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Callable[[EverythingCallback], None]) -> int: ...
@overload
def f2(x: Callable[[Callable[[], None]], None]) -> str: ...
def f2(x: Any) -> int | str: ...

@overload
def f3(x: Callable[..., None]) -> int: ...
@overload
def f3(x: Callable[[], None]) -> str: ...
def f3(x: Any) -> int | str: ...

def g(x: Callable[..., None], y: Callable[[Callable[..., None]], None]):
    assert_type(f1(x), Any)
    assert_type(f2(y), int)
    assert_type(f3(x), int)
    "#,
);

testcase!(
    test_list_vs_sequence_materialization,
    r#"
from typing import Any, assert_type, overload, Sequence

@overload
def f1(x: list[object]) -> int: ...
@overload
def f1(x: list[Any]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: Sequence[object]) -> int: ...
@overload
def f2(x: Sequence[Any]) -> str: ...
def f2(x: Any) -> int | str: ...

def g(x: list[Any]):
    assert_type(f1(x), Any)
    assert_type(f2(x), int)
    "#,
);

testcase!(
    test_tuple_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(x: tuple[Any, ...]) -> int: ...
@overload
def f1(x: tuple[()]) -> str: ...
def f1(x: Any) -> int | str: ...

@overload
def f2(x: tuple[int, ...]) -> int: ...
@overload
def f2(x: tuple[str, ...]) -> str: ...
def f2(x: Any) -> int | str: ...

def g(x: tuple[Any, ...]):
    assert_type(f1(x), int)
    assert_type(f2(x), Any)
    "#,
);

testcase!(
    test_abstractmethod_does_not_need_implementation,
    r#"
from typing import overload
from abc import ABC, abstractmethod

class A(ABC):
    @overload
    @abstractmethod
    def f(self, x: int) -> int: ...
    @overload
    @abstractmethod
    def f(self, x: str) -> str: ...

class B(ABC):
    @abstractmethod
    @overload
    def f(self, x: int) -> int: ...
    @abstractmethod
    @overload
    def f(Self, x: str) -> str: ...
    "#,
);

testcase!(
    test_overload_error_shows_argument_types,
    r#"
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x): return x

# Test with wrong type - should show argument type in error
f(3.14)  # E: No matching overload found for function `f` called with arguments: (float)

# Test with keyword argument
f(x=3.14)  # E: No matching overload found for function `f` called with arguments: (x=float)

# Test with multiple arguments (Pyrefly infers literal types for constants)
@overload
def g(x: int, y: int) -> int: ...
@overload
def g(x: str, y: str) -> str: ...
def g(x, y): return x

g(1, "hello")  # E: No matching overload found for function `g` called with arguments: (Literal[1], Literal['hello'])
g(x=1, y="hello")  # E: No matching overload found for function `g` called with arguments: (x=Literal[1], y=Literal['hello'])
    "#,
);

testcase!(
    test_varargs_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(*args: int) -> int: ...
@overload
def f1(*args: str) -> str: ...
def f1(*args) -> int | str: ...

@overload
def f2(*args: Any) -> int: ...
@overload
def f2(*args: str) -> str: ...
def f2(*args) -> int | str: ...

def g(*args):
    # ambiguous whether this matches `f1`'s first or second overload, so fall back to return type Any
    assert_type(f1(*args), Any)
    # this matches `f2`'s first overload via https://typing.python.org/en/latest/spec/overload.html#step-5
    assert_type(f2(*args), int)

def h(args: Any):
    # make sure the entire iterable being `Any` works
    assert_type(f1(*args), Any)
    assert_type(f2(*args), int)
    "#,
);

testcase!(
    test_kwargs_materialization,
    r#"
from typing import Any, assert_type, overload

@overload
def f1(**kwargs: int) -> int: ...
@overload
def f1(**kwargs: str) -> str: ...
def f1(**kwargs) -> int | str: ...

@overload
def f2(**kwargs: Any) -> int: ...
@overload
def f2(**kwargs: str) -> str: ...
def f2(**kwargs) -> int | str: ...

def g(**kwargs):
    # ambiguous whether this matches `f1`'s first or second overload, so fall back to return type Any
    assert_type(f1(**kwargs), Any)
    # this matches `f2`'s first overload via https://typing.python.org/en/latest/spec/overload.html#step-5
    assert_type(f2(**kwargs), int)

def h(kwargs: Any):
    # make sure the entire mapping being `Any` works
    assert_type(f1(**kwargs), Any)
    assert_type(f2(**kwargs), int)
    "#,
);

testcase!(
    test_unsolved_typevar,
    r#"
from typing import overload
class Foo[T]:
    @overload
    def test(self, obj: None, cls: type[T]) -> str: ...
    @overload
    def test(self, obj: T, cls: type[T]) -> int: ...
    def test(self, obj: T | None, cls: type[T]) -> str | int: ...
    "#,
);
