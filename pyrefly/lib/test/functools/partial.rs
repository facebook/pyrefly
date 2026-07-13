/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.partial` conformance: argument validation on plain functions, class objects,
//! and callable targets.
//!
//! pyrefly has no native `partial` modeling (it relies on the typeshed stub whose
//! `__call__(*args: Any, **kwargs: Any) -> _T` erases argument info), so it gets the return
//! type right but validates no arguments. Divergences are `bug=`-marked; the correct
//! (runtime) behavior is recorded inline as `# WANT: ...`. To flip a test when native
//! support lands: drop `bug=` and turn each `# WANT: X` into `# E: X` (or delete a now-spurious
//! `# E:`). See `partial_generics.rs`, `partial_edge.rs`, and `generic_basic.rs`'s
//! `test_functools_partial_pattern`.

use crate::functools_testcase;

// ===== Basic: bind nothing / positional / keyword =====

functools_testcase!(
    test_partial_basic_no_bind,
    r#"
from typing import reveal_type
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p1 = functools.partial(foo)
p1(1, "a", 3)
p1(1, "a", c=3)
p1(1, b="a", c=3)
reveal_type(p1)  # E: revealed type: partial[int]
"#,
);

functools_testcase!(
    test_partial_basic_callable_compat,
    r#"
from typing import Callable
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p1 = functools.partial(foo)
def takes_callable_int(f: Callable[..., int]) -> None: ...
def takes_callable_str(f: Callable[..., str]) -> None: ...
takes_callable_int(p1)
takes_callable_str(p1)  # E: Argument `partial[int]` is not assignable to parameter `f` with type `(...) -> str` in function `takes_callable_str`
"#,
);

functools_testcase!(
    bug = "partial does not type-check remaining args at call time; the int-for-str and arity/keyword errors are missed",
    test_partial_basic_one_positional,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p2 = functools.partial(foo, 1)
p2("a")
p2("a", 3)
p2("a", c=3)
p2(1, 3)  # WANT: Argument 1 to "foo" has incompatible type "int"; expected "str"
p2(1, "a", 3)  # WANT: Too many arguments for "foo" / Argument 1 incompatible / Argument 2 incompatible
p2(a=1, b="a", c=3)  # WANT: Unexpected keyword argument "a" for "foo"
"#,
);

functools_testcase!(
    bug = "partial with a keyword-bound arg does not validate the remaining args at call time",
    test_partial_basic_keyword_bind,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p3 = functools.partial(foo, b="a")
p3(1)
p3(1, c=3)
p3(a=1)
p3(1, b="a", c=3)  # OK, keywords can be clobbered
p3(1, 3)  # WANT: Too many positional arguments for "foo" / Argument 2 incompatible
"#,
);

functools_testcase!(
    bug = "partial does not type-check bound args at construction time; only the non-callable target is caught",
    test_partial_basic_construct_arg_check,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
functools.partial(foo, "a")  # WANT: Argument 1 to "foo" has incompatible type "str"; expected "int"
functools.partial(foo, b=1)  # WANT: Argument "b" to "foo" has incompatible type "int"; expected "str"
functools.partial(foo, a=1, b=2, c=3)  # WANT: Argument "b" to "foo" has incompatible type "int"; expected "str"
functools.partial(1)  # E: Argument `Literal[1]` is not assignable to parameter `func` with type `(...) -> @_` in function `functools.partial.__new__`
"#,
);

// ===== Star: *args / **kwargs / keyword-only targets =====

functools_testcase!(
    bug = "partial of a function with *args/**kwargs does not check the bad *args element or **kwargs value at call time",
    test_partial_star_bound_prefix,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p1 = functools.partial(foo, 1, d="a", x=9)
p1("a", 2, 3, 4)
p1("a", 2, 3, 4, d="a")
p1("a", 2, 3, 4, "a")  # WANT: Argument 5 to "foo" has incompatible type "str"; expected "int"
p1("a", 2, 3, 4, x="a")  # WANT: Argument "x" to "foo" has incompatible type "str"; expected "int"
"#,
);

functools_testcase!(
    bug = "partial does not report the missing keyword-only `d` (and the bad positional) when binding a prefix",
    test_partial_star_bound_two,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p2 = functools.partial(foo, 1, "a")
p2(2, 3, 4, d="a")
p2("a")  # WANT: Missing named argument "d" for "foo" # WANT: Argument 1 to "foo" has incompatible type "str"; expected "int"
p2(2, 3, 4)  # WANT: Missing named argument "d" for "foo"
"#,
);

functools_testcase!(
    bug = "partial does not check bound *args element types at construction time",
    test_partial_star_construct,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
functools.partial(foo, 1, "a", "b", "c", d="a")  # WANT: Argument 3 to "foo" has incompatible type "str"; expected "int" # WANT: Argument 4 to "foo" has incompatible type "str"; expected "int"
"#,
);

functools_testcase!(
    bug = "partial does not validate **-unpacked dict args against the wrapped *args/**kwargs element types",
    test_partial_star_unpack,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p1 = functools.partial(foo, 1, d="a", x=9)
def bar(*a: bytes, **k: int) -> None:
    p1("a", 2, 3, 4, d="a", **k)
    p1("a", d="a", **k)
    p1("a", **k)  # WANT: Argument 2 to "foo" has incompatible type "**dict[str, int]"; expected "str"
    p1(**k)  # WANT: Argument 1 to "foo" has incompatible type "**dict[str, int]"; expected "str"
    p1(*a)  # WANT: Expected iterable as variadic argument
"#,
);

functools_testcase!(
    bug = "partial(baz, *xs) does not track the consumed positionals, so the over-arity call is not flagged",
    test_partial_star_iterable,
    r#"
import functools
from typing import List
def baz(a: int, b: int) -> int: ...
def test_baz(xs: List[int]) -> None:
    p3 = functools.partial(baz, *xs)
    p3()
    p3(1)  # WANT: Too many arguments for "baz"
"#,
);

// ===== Callable / protocol targets =====

functools_testcase!(
    bug = "partial of a Callable parameter does not check the remaining arg type, nor an unexpected bound keyword",
    test_partial_callable_plain,
    r#"
from typing import Callable
import functools
def main1(f: Callable[[int, str], int]) -> None:
    p = functools.partial(f, 1)
    p("a")
    p(1)  # WANT: Argument 1 has incompatible type "int"; expected "str"
    functools.partial(f, a=1)  # WANT: Unexpected keyword argument "a"
"#,
);

functools_testcase!(
    bug = "partial of a callback protocol does not check the remaining positional against __call__'s signature",
    test_partial_callable_protocol,
    r#"
import functools
class CallbackProto:
    def __call__(self, a: int, b: str) -> int: ...
def main2(f: CallbackProto) -> None:
    p = functools.partial(f, b="a")
    p(1)
    p("a")  # WANT: Argument 1 to "__call__" of "CallbackProto" has incompatible type "str"; expected "int"
"#,
);

// ===== Class-object (Type[...]) targets =====

functools_testcase!(
    bug = "partial over a class does not check bound/remaining args against __init__",
    test_partial_type_class,
    r#"
import functools
from typing import reveal_type
class A:
    def __init__(self, a: int, b: str) -> None: ...
p = functools.partial(A, 1)
reveal_type(p)  # E: revealed type: partial[A]
p("a")
p(1)  # WANT: Argument 1 to "A" has incompatible type "int"; expected "str"
p(z=1)  # WANT: Unexpected keyword argument "z" for "A"
"#,
);

functools_testcase!(
    bug = "partial over a `Type[A]` value does not check bound/remaining args against __init__",
    test_partial_type_type_of,
    r#"
import functools
from typing import Type, reveal_type
class A:
    def __init__(self, a: int, b: str) -> None: ...
def main(t: Type[A]) -> None:
    p = functools.partial(t, 1)
    reveal_type(p)  # E: revealed type: partial[A]
    p("a")
    p(1)  # WANT: Argument 1 to "A" has incompatible type "int"; expected "str"
    p(z=1)  # WANT: Unexpected keyword argument "z" for "A"
"#,
);

functools_testcase!(
    bug = "partial over a concrete class object does not check the bound arg against __init__",
    test_partial_type_object_plain,
    r#"
import functools
from typing import Type, reveal_type
class A:
    def __init__(self, val: int) -> None: ...
def f1(cls1: Type[A]) -> None:
    reveal_type(functools.partial(cls1, 2)())  # E: revealed type: A
    functools.partial(cls1, "asdf")  # WANT: Argument 1 to "A" has incompatible type "str"; expected "int"
"#,
);

functools_testcase!(
    bug = "partial over a parameterized generic class object does not check the bound arg against the specialized __init__",
    test_partial_type_object_generic,
    r#"
import functools
from typing import Type, Generic, TypeVar, reveal_type
T = TypeVar("T")
class B(Generic[T]):
    def __init__(self, val: T) -> None: ...
def f2(cls2: Type[B[int]]) -> None:
    reveal_type(functools.partial(cls2, 2)())  # E: revealed type: B[int]
    functools.partial(cls2, "asdf")  # WANT: Argument 1 to "B" has incompatible type "str"; expected "int"
"#,
);

functools_testcase!(
    bug = "partial over a `Type[B[T]]` does not check the bound arg against the unsolved type parameter T",
    test_partial_type_object_generic_param,
    r#"
import functools
from typing import Type, Generic, TypeVar, reveal_type
T = TypeVar("T")
class B(Generic[T]):
    def __init__(self, val: T) -> None: ...
def foo(cls3: Type[B[T]]) -> None:
    # WANT: Argument 1 to "B" has incompatible type "str"; expected "T"
    reveal_type(functools.partial(cls3, "asdf"))  # E: revealed type: partial[B[T]]
    # WANT: Argument 1 to "B" has incompatible type "int"; expected "T"
    reveal_type(functools.partial(cls3, 2)())  # E: revealed type: B[T]
"#,
);

// ===== Union targets =====

functools_testcase!(
    bug = "partial over a union that contains a non-callable (str) misses the \"str not callable\" error",
    test_partial_union_with_noncallable,
    r#"
import functools
from typing import Any, Callable, Union, reveal_type
def f(
    cls1: Any,
    cls2: Union[Any, Any],
    fn1: Union[Callable[[int], int], Callable[[int], int]],
    fn2: Union[Callable[[int], int], Callable[[int], str]],
    fn3: Union[Callable[[int], int], str],
) -> None:
    reveal_type(functools.partial(cls1, 2)())  # E: revealed type: Any
    reveal_type(functools.partial(cls2, 2)())  # E: revealed type: Any
    reveal_type(functools.partial(fn1, 2)())  # E: revealed type: int
    reveal_type(functools.partial(fn2, 2)())  # E: revealed type: int | str
    # WANT: also emit `"str" not callable`
    reveal_type(functools.partial(fn3, 2)())  # E: revealed type: int # E: Argument `((int) -> int) | str` is not assignable to parameter `func` with type `(...) -> int` in function `functools.partial.__new__`
"#,
);

functools_testcase!(
    test_partial_union_class_or_factory,
    r#"
import functools
from typing import Callable, Union, Type, reveal_type
from typing_extensions import TypeAlias
class FooBar:
    def __init__(self, arg1: str) -> None:
        pass
def f1(t: Union[Type[FooBar], Callable[..., 'FooBar']]) -> None:
    val = functools.partial(t)
    reveal_type(val)  # E: revealed type: partial[FooBar]
FooBarFunc: TypeAlias = Callable[..., 'FooBar']
def f2(t: Union[Type[FooBar], FooBarFunc]) -> None:
    val = functools.partial(t)
    reveal_type(val)  # E: revealed type: partial[FooBar]
"#,
);

// ===== TypedDict Unpack **kwargs =====

functools_testcase!(
    bug = "partial with TypedDict-Unpack **kwargs does not validate later call kwargs: bad type and unexpected key are missed",
    test_partial_typeddict_fn1_positional,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D1(TypedDict, total=False):
    a1: int
def fn1(a1: int) -> None: ...
def main1(**d1: Unpack[D1]) -> None:
    partial(fn1, **d1)()
    partial(fn1, **d1)(**d1)
    partial(fn1, **d1)(a1=1)
    partial(fn1, **d1)(a1="asdf")  # WANT: Argument "a1" to "fn1" has incompatible type "str"; expected "int"
    partial(fn1, **d1)(oops=1)  # WANT: Unexpected keyword argument "oops" for "fn1"
"#,
);

functools_testcase!(
    bug = "partial of a function whose **kwargs is a TypedDict Unpack does not validate later call kwargs",
    test_partial_typeddict_fn2_kwargs,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D1(TypedDict, total=False):
    a1: int
def fn2(**kwargs: Unpack[D1]) -> None: ...
def main2(**d1: Unpack[D1]) -> None:
    partial(fn2, **d1)()
    partial(fn2, **d1)(**d1)
    partial(fn2, **d1)(a1=1)
    partial(fn2, **d1)(a1="asdf")  # WANT: Argument "a1" to "fn2" has incompatible type "str"; expected "int"
    partial(fn2, **d1)(oops=1)  # WANT: Unexpected keyword argument "oops" for "fn2"
"#,
);

functools_testcase!(
    bug = "partial with a partial TypedDict Unpack prefix does not validate the remaining required/typed kwargs",
    test_partial_typeddict_fn3_mixed,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D2(TypedDict, total=False):
    a1: int
    a2: str
class A2Good(TypedDict, total=False):
    a2: str
class A2Bad(TypedDict, total=False):
    a2: int
def fn3(a1: int, a2: str) -> None: ...
def main3(a2good: A2Good, a2bad: A2Bad, **d2: Unpack[D2]) -> None:
    partial(fn3, **d2)()
    partial(fn3, **d2)(a1=1, a2="asdf")
    partial(fn3, **d2)(**d2)
    partial(fn3, **d2)(a1="asdf")  # WANT: Argument "a1" to "fn3" has incompatible type "str"; expected "int"
    partial(fn3, **d2)(a1=1, a2="asdf", oops=1)  # WANT: Unexpected keyword argument "oops" for "fn3"
    partial(fn3, **d2)(**a2good)
    partial(fn3, **d2)(**a2bad)  # WANT: Argument "a2" to "fn3" has incompatible type "int"; expected "str"
"#,
);

functools_testcase!(
    bug = "partial of a **kwargs-Unpack function does not validate the remaining kwargs supplied at call time",
    test_partial_typeddict_fn4_kwargs_mixed,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D2(TypedDict, total=False):
    a1: int
    a2: str
class A2Good(TypedDict, total=False):
    a2: str
class A2Bad(TypedDict, total=False):
    a2: int
def fn3(a1: int, a2: str) -> None: ...
def fn4(**kwargs: Unpack[D2]) -> None: ...
def main4(a2good: A2Good, a2bad: A2Bad, **d2: Unpack[D2]) -> None:
    partial(fn4, **d2)()
    partial(fn4, **d2)(a1=1, a2="asdf")
    partial(fn4, **d2)(**d2)
    partial(fn4, **d2)(a1="asdf")  # WANT: Argument "a1" to "fn4" has incompatible type "str"; expected "int"
    partial(fn4, **d2)(a1=1, a2="asdf", oops=1)  # WANT: Unexpected keyword argument "oops" for "fn4"
    partial(fn3, **d2)(**a2good)
    partial(fn3, **d2)(**a2bad)  # WANT: Argument "a2" to "fn3" has incompatible type "int"; expected "str"
"#,
);

functools_testcase!(
    bug = "partial does not flag a TypedDict-Unpack prefix that supplies a key the target does not accept",
    test_partial_typeddict_extra_key,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D1(TypedDict, total=False):
    a1: int
class D2(TypedDict, total=False):
    a1: int
    a2: str
def fn1(a1: int) -> None: ...
def fn2(**kwargs: Unpack[D1]) -> None: ...
def main5(**d2: Unpack[D2]) -> None:
    partial(fn1, **d2)()  # WANT: Extra argument "a2" from **args for "fn1"
    partial(fn2, **d2)()  # WANT: Extra argument "a2" from **args for "fn2"
"#,
);

functools_testcase!(
    bug = "partial with a too-narrow TypedDict-Unpack prefix does not report the missing/too-many/bad positionals at call time",
    test_partial_typeddict_missing,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D1(TypedDict, total=False):
    a1: int
class A2Good(TypedDict, total=False):
    a2: str
class A2Bad(TypedDict, total=False):
    a2: int
def fn3(a1: int, a2: str) -> None: ...
def fn4(**kwargs) -> None: ...
def main6(a2good: A2Good, a2bad: A2Bad, **d1: Unpack[D1]) -> None:
    partial(fn3, **d1)()  # WANT: Missing positional argument "a1" in call to "fn3"
    partial(fn3, **d1)("asdf")  # WANT: Too many positional arguments / Too few arguments / Argument 1 incompatible
    partial(fn3, **d1)(a2="asdf")
    partial(fn3, **d1)(**a2good)
    partial(fn3, **d1)(**a2bad)  # WANT: Argument "a2" to "fn3" has incompatible type "int"; expected "str"
    partial(fn4, **d1)()
    partial(fn4, **d1)("asdf")
    partial(fn4, **d1)(a2="asdf")
    partial(fn4, **d1)(**a2good)
    partial(fn4, **d1)(**a2bad)
"#,
);

// ===== Misc single scenarios =====

functools_testcase!(
    bug = "partial over a TypeGuard function should reveal partial[bool], but pyrefly keeps partial[TypeGuard[list[str]]]",
    test_partial_wrapping_type_guard,
    r#"
from typing import reveal_type
import functools
from typing_extensions import TypeGuard
def is_str_list(val: list[object]) -> TypeGuard[list[str]]: ...
# WANT: revealed type: partial[bool]
reveal_type(functools.partial(is_str_list, [1, 2, 3]))  # E: revealed type: partial[TypeGuard[list[str]]]
reveal_type(functools.partial(is_str_list, [1, 2, 3])())  # E: revealed type: bool
"#,
);

functools_testcase!(
    bug = "partial with a TypeVarTuple callable doesn't check argument compatibility; the mismatched call is not flagged",
    test_partial_type_var_tuple_callable,
    r#"
import functools
import typing
Ts = typing.TypeVarTuple("Ts")
def foo(fn: typing.Callable[[typing.Unpack[Ts]], None], /, *arg: typing.Unpack[Ts], kwarg: str) -> None: ...
p = functools.partial(foo, kwarg="asdf")
def bar(a: int, b: str, c: float) -> None: ...
p(bar, 1, "a", 3.0)
p(bar, 1, "a", 3.0, kwarg="asdf")
p(bar, 1, "a", "b")  # WANT: Argument 1 to "foo" has incompatible type "Callable[[int, str, float], None]"; expected "Callable[[int, str, str], None]"
"#,
);

functools_testcase!(
    bug = "nested partial(partial, ...) loses all type info: reveal_type yields Unknown instead of int and bad calls are not reported",
    test_partial_of_partial,
    r#"
from typing import reveal_type
from functools import partial
def foo(x: int) -> int: ...
p = partial(partial, foo)
# WANT: revealed type: int
reveal_type(p()(1))  # E: revealed type: Unknown
p()("no")  # WANT: Argument 1 to "foo" has incompatible type "str"; expected "int"
q = partial(partial, partial, foo)
q()()("no")  # WANT: Argument 1 to "foo" has incompatible type "str"; expected "int"
r = partial(partial, foo, 1)
# WANT: revealed type: int
reveal_type(r()())  # E: revealed type: Unknown
"#,
);

functools_testcase!(
    bug = "partial(fn, 1) is (str, bytes) -> int; passing it where Callable[[str, int], int] is expected should error (known false negative)",
    test_partial_as_callable_arg_mismatch,
    r#"
from functools import partial
from typing import Callable
def fn(a: int, b: str, c: bytes) -> int: ...
def callback1(fn: Callable[[str, bytes], int]) -> None: ...
def callback2(fn: Callable[[str, int], int]) -> None: ...
callback1(partial(fn, 1))
callback2(partial(fn, 1))  # WANT: Argument has incompatible type "partial[int]"; expected "Callable[[str, int], int]"
"#,
);

functools_testcase!(
    bug = "partial wrapping a class object does not check argument types; the incompatible str arg is missed",
    test_partial_class_object_arg_check,
    r#"
from typing import reveal_type
from functools import partial
class A:
    def __init__(self, var: int, b: int, c: int) -> None: ...
p = partial(A, 1)
reveal_type(p)  # E: revealed type: partial[A]
p(1, "no")  # WANT: Argument 2 to "A" has incompatible type "str"; expected "int"
q: partial[A] = partial(A, 1)
"#,
);

functools_testcase!(
    bug = "partial wrapping an abstract class is not flagged: partial(A) and the resulting call should both error, but only the direct A() is caught",
    test_partial_abstract_class,
    r#"
from abc import ABC, abstractmethod
from functools import partial
class A(ABC):
    def __init__(self) -> None: ...
    @abstractmethod
    def method(self) -> None: ...
def f1(cls: type[A]) -> None:
    cls()
    partial_cls = partial(cls)
    partial_cls()
def f2() -> None:
    A()  # E: Cannot instantiate `A` because the following members are abstract: `method`
    partial_cls = partial(A)  # WANT: Cannot instantiate abstract class "A" with abstract attribute "method"
    partial_cls()  # WANT: Cannot instantiate abstract class "A" with abstract attribute "method"
"#,
);

functools_testcase!(
    test_partial_classmethod_returns_self,
    r#"
from functools import partial
from typing_extensions import Self
class A:
    def __init__(self, ts: float, msg: str) -> None: ...
    @classmethod
    def from_msg(cls, msg: str) -> Self:
        factory = partial(cls, ts=0)
        return factory(msg=msg)
"#,
);
