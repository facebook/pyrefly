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
use crate::test::util::TestEnv;
use crate::testcase;

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
reveal_type(p1)  # E: revealed type: (a: int, b: str, c: int = 5) -> int
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
takes_callable_str(p1)  # E: Argument `(a: int, b: str, c: int = 5) -> int` is not assignable to parameter `f` with type `(...) -> str` in function `takes_callable_str`
"#,
);

functools_testcase!(
    test_partial_basic_one_positional,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p2 = functools.partial(foo, 1)
p2("a")
p2("a", 3)
p2("a", c=3)
p2(1, 3)  # E: Argument `Literal[1]` is not assignable to parameter `b` with type `str`
p2(1, "a", 3)  # E: Argument `Literal[1]` is not assignable to parameter `b` with type `str` # E: Argument `Literal['a']` is not assignable to parameter `c` with type `int` # E: Expected 2 positional arguments, got 3
p2(a=1, b="a", c=3)  # E: Unexpected keyword argument `a`
"#,
);

functools_testcase!(
    test_partial_basic_keyword_bind,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
p3 = functools.partial(foo, b="a")
p3(1)
p3(1, c=3)
p3(a=1)
p3(1, b="a", c=3)  # OK, keywords can be clobbered
p3(1, 3)  # E: Expected 1 positional argument, got 2
"#,
);

functools_testcase!(
    test_partial_basic_construct_arg_check,
    r#"
import functools
def foo(a: int, b: str, c: int = 5) -> int: ...
functools.partial(foo, "a")  # E: Argument `Literal['a']` is not assignable to parameter `a` with type `int` in function `foo`
functools.partial(foo, b=1)  # E: Argument `Literal[1]` is not assignable to parameter `b` with type `str` in function `foo`
functools.partial(foo, a=1, b=2, c=3)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str` in function `foo`
functools.partial(1)  # E: Argument `Literal[1]` is not assignable to parameter `func` with type `(...) -> @_` in function `functools.partial.__new__`
"#,
);

// ===== Star: *args / **kwargs / keyword-only targets =====

functools_testcase!(
    test_partial_star_bound_prefix,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p1 = functools.partial(foo, 1, d="a", x=9)
p1("a", 2, 3, 4)
p1("a", 2, 3, 4, d="a")
p1("a", 2, 3, 4, "a")  # E: Argument `Literal['a']` is not assignable to parameter `*args` with type `int`
p1("a", 2, 3, 4, x="a")  # E: Keyword argument `x` with type `Literal['a']` is not assignable to parameter `**kwargs` with type `int`
"#,
);

functools_testcase!(
    test_partial_star_bound_two,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p2 = functools.partial(foo, 1, "a")
p2(2, 3, 4, d="a")
p2("a")  # E: Missing argument `d` # E: Argument `Literal['a']` is not assignable to parameter `*args` with type `int`
p2(2, 3, 4)  # E: Missing argument `d`
"#,
);

functools_testcase!(
    test_partial_star_construct,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
functools.partial(foo, 1, "a", "b", "c", d="a")  # E: Argument `Literal['b']` is not assignable to parameter `*args` with type `int` in function `foo` # E: Argument `Literal['c']` is not assignable to parameter `*args` with type `int` in function `foo`
"#,
);

functools_testcase!(
    test_partial_star_unpack,
    r#"
import functools
def foo(a: int, b: str, *args: int, d: str, **kwargs: int) -> int: ...
p1 = functools.partial(foo, 1, d="a", x=9)
def bar(*a: bytes, **k: int) -> None:
    p1("a", 2, 3, 4, d="a", **k)
    p1("a", d="a", **k)
    p1("a", **k)  # E: Unpacked keyword argument `int` is not assignable to parameter `d` with type `str`
    p1(**k)  # E: Unpacked keyword argument `int` is not assignable to parameter `b` with type `str` # E: Unpacked keyword argument `int` is not assignable to parameter `d` with type `str`
    p1(*a)  # E: Argument `bytes` is not assignable to parameter `b` with type `str` # E: Argument `bytes` is not assignable to parameter `*args` with type `int`
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
    test_partial_callable_plain,
    r#"
from typing import Callable
import functools
def main1(f: Callable[[int, str], int]) -> None:
    p = functools.partial(f, 1)
    p("a")
    p(1)  # E: Argument `Literal[1]` is not assignable to parameter with type `str`
    functools.partial(f, a=1)  # E: Unexpected keyword argument `a`
"#,
);

functools_testcase!(
    test_partial_callable_protocol,
    r#"
import functools
class CallbackProto:
    def __call__(self, a: int, b: str) -> int: ...
def main2(f: CallbackProto) -> None:
    p = functools.partial(f, b="a")
    p(1)
    p("a")  # E: Argument `Literal['a']` is not assignable to parameter `a` with type `int`
"#,
);

// Cross-module: a callback protocol imported across the module boundary still resolves its
// `__call__`, so the residual is argument-checked.
testcase!(
    test_partial_callback_protocol_cross_module,
    TestEnv::one(
        "cbmod",
        "class Cb:\n    def __call__(self, a: int, b: str) -> int: ...\n",
    ),
    r#"
import functools
from cbmod import Cb
def m(f: Cb) -> None:
    p = functools.partial(f, b="a")
    p(1)
    p("a")  # E: Argument `Literal['a']` is not assignable to parameter `a` with type `int`
"#,
);

// ===== Nominal `partial` attributes =====

functools_testcase!(
    test_partial_attribute_access,
    r#"
import functools
from typing import reveal_type
def f(a: int, b: str, c: float) -> bytes: return b""
p = functools.partial(f, 1)
reveal_type(p.func)  # E: revealed type: (...) -> bytes
reveal_type(p.args)  # E: revealed type: tuple[Any, ...]
reveal_type(p.keywords)  # E: revealed type: dict[str, Any]
x: object = p
if isinstance(x, functools.partial):
    reveal_type(x)  # E: revealed type: partial[Unknown]
"#,
);

// ===== Class-object (Type[...]) targets =====

functools_testcase!(
    test_partial_type_class,
    r#"
import functools
from typing import reveal_type
class A:
    def __init__(self, a: int, b: str) -> None: ...
p = functools.partial(A, 1)
reveal_type(p)  # E: revealed type: (b: str) -> A
p("a")
p(1)  # E: Argument `Literal[1]` is not assignable to parameter `b` with type `str`
p(z=1)  # E: Missing argument `b` # E: Unexpected keyword argument `z`
"#,
);

functools_testcase!(
    test_partial_type_type_of,
    r#"
import functools
from typing import Type, reveal_type
class A:
    def __init__(self, a: int, b: str) -> None: ...
def main(t: Type[A]) -> None:
    p = functools.partial(t, 1)
    reveal_type(p)  # E: revealed type: (b: str) -> A
    p("a")
    p(1)  # E: Argument `Literal[1]` is not assignable to parameter `b` with type `str`
    p(z=1)  # E: Missing argument `b` # E: Unexpected keyword argument `z`
"#,
);

functools_testcase!(
    test_partial_type_object_plain,
    r#"
import functools
from typing import Type, reveal_type
class A:
    def __init__(self, val: int) -> None: ...
def f1(cls1: Type[A]) -> None:
    reveal_type(functools.partial(cls1, 2)())  # E: revealed type: A
    functools.partial(cls1, "asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `val` with type `int` in function `A.__init__`
"#,
);

functools_testcase!(
    test_partial_type_object_generic,
    r#"
import functools
from typing import Type, Generic, TypeVar, reveal_type
T = TypeVar("T")
class B(Generic[T]):
    def __init__(self, val: T) -> None: ...
def f2(cls2: Type[B[int]]) -> None:
    reveal_type(functools.partial(cls2, 2)())  # E: revealed type: B[int]
    functools.partial(cls2, "asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `val` with type `int` in function `B.__init__`
"#,
);

functools_testcase!(
    test_partial_type_object_generic_param,
    r#"
import functools
from typing import Type, Generic, TypeVar, reveal_type
T = TypeVar("T")
class B(Generic[T]):
    def __init__(self, val: T) -> None: ...
def foo(cls3: Type[B[T]]) -> None:
    reveal_type(functools.partial(cls3, "asdf"))  # E: revealed type: () -> B[T] # E: Argument `Literal['asdf']` is not assignable to parameter `val` with type `T` in function `B.__init__`
    reveal_type(functools.partial(cls3, 2)())  # E: revealed type: B[T] # E: Argument `Literal[2]` is not assignable to parameter `val` with type `T` in function `B.__init__`
"#,
);

// ===== Union targets =====

functools_testcase!(
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
    reveal_type(functools.partial(fn3, 2)())  # E: revealed type: int # E: Expected a callable, got `str` # E: Argument `((int) -> int) | str` is not assignable to parameter `func` with type `(...) -> int` in function `functools.partial.__new__`
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

// Binding a `**` splat of an `Unpack[TypedDict]` consumes exactly the TypedDict's declared fields,
// so the residual drops those parameters and the remaining ones are checked on the later call.
functools_testcase!(
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
    partial(fn1, **d1)(a1="asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `a1` with type `int`
    partial(fn1, **d1)(oops=1)  # E: Unexpected keyword argument `oops`
"#,
);

functools_testcase!(
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
    partial(fn2, **d1)(a1="asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `a1` with type `int`
    partial(fn2, **d1)(oops=1)  # E: Unexpected keyword argument `oops`
"#,
);

functools_testcase!(
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
    partial(fn3, **d2)(a1="asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `a1` with type `int`
    partial(fn3, **d2)(a1=1, a2="asdf", oops=1)  # E: Unexpected keyword argument `oops`
    partial(fn3, **d2)(**a2good)
    partial(fn3, **d2)(**a2bad)  # E: Argument `int` is not assignable to parameter `a2` with type `str`
"#,
);

functools_testcase!(
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
    partial(fn4, **d2)(a1="asdf")  # E: Argument `Literal['asdf']` is not assignable to parameter `a1` with type `int`
    partial(fn4, **d2)(a1=1, a2="asdf", oops=1)  # E: Unexpected keyword argument `oops`
    partial(fn3, **d2)(**a2good)
    partial(fn3, **d2)(**a2bad)  # E: Argument `int` is not assignable to parameter `a2` with type `str`
"#,
);

functools_testcase!(
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
    partial(fn1, **d2)()  # E: Unexpected keyword argument `a2` in function `fn1`
    partial(fn2, **d2)()  # E: Unexpected keyword argument `a2` in function `fn2`
"#,
);

functools_testcase!(
    bug = "an optional TypedDict-Unpack prefix key is treated as always-bound, so a missing/positional diagnostic names the wrong parameter compared to mypy",
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
    # `a1` is bound optionally from `**d1`, but `a2` is never bound, so `a2` is the parameter
    # flagged as missing (mypy also flags the still-optional `a1`).
    partial(fn3, **d1)()  # E: Missing argument `a2`
    partial(fn3, **d1)("asdf")  # E: Expected argument `a2` to be passed by name
    partial(fn3, **d1)(a2="asdf")
    partial(fn3, **d1)(**a2good)
    partial(fn3, **d1)(**a2bad)  # E: Argument `int` is not assignable to parameter `a2` with type `str`
    partial(fn4, **d1)()
    partial(fn4, **d1)("asdf")  # E: Expected 0 positional arguments, got 1
    partial(fn4, **d1)(a2="asdf")
    partial(fn4, **d1)(**a2good)
    partial(fn4, **d1)(**a2bad)
"#,
);

// Expansion covers inherited TypedDict fields, so binding a subclass splat consumes the base-class
// key and the residual validates it on the later call.
functools_testcase!(
    test_partial_typeddict_inherited_field,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class Base(TypedDict):
    a: int
class Sub(Base, total=False):
    b: str
def fn(a: int, b: str) -> None: ...
def main(**d: Unpack[Sub]) -> None:
    partial(fn, **d)(a=1, b="x")
    partial(fn, **d)(a="no")  # E: Argument `Literal['no']` is not assignable to parameter `a` with type `int`
    partial(fn, **d)(oops=1)  # E: Unexpected keyword argument `oops`
"#,
);

// Binding a required field of a `**kwargs: Unpack[TypedDict]` target satisfies it: the later call
// need not re-supply it, but an unbound required field is still enforced.
functools_testcase!(
    test_partial_typeddict_required_field_bound,
    r#"
from typing import TypedDict
from typing_extensions import Unpack
from functools import partial
class D(TypedDict):
    a1: int
    a2: str
def fn(**kwargs: Unpack[D]) -> None: ...
partial(fn)
partial(fn)()  # E: Missing argument `a1` # E: Missing argument `a2`
partial(fn, a1=1)
partial(fn, a1=1)()  # E: Missing argument `a2`
partial(fn, a1=1, a2="x")()
partial(fn, a1="no")  # E: Argument `Literal['no']` is not assignable to parameter `a1` with type `int` in function `fn`
"#,
);

// ===== Misc single scenarios =====

functools_testcase!(
    test_partial_wrapping_type_guard,
    r#"
from typing import reveal_type
import functools
from typing_extensions import TypeGuard
def is_str_list(val: list[object]) -> TypeGuard[list[str]]: ...
reveal_type(functools.partial(is_str_list, [1, 2, 3]))  # E: revealed type: () -> bool
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

// Default mode is gradual on the residual as a subtype, so a `Callable`-target param mismatch is
// not reported. The mismatch fires only under the flag (the `_strict` twin below).
testcase!(
    test_partial_as_callable_arg_mismatch,
    TestEnv::new(),
    r#"
from functools import partial
from typing import Callable
def fn(a: int, b: str, c: bytes) -> int: ...
def callback1(fn: Callable[[str, bytes], int]) -> None: ...
def callback2(fn: Callable[[str, int], int]) -> None: ...
callback1(partial(fn, 1))
callback2(partial(fn, 1))
"#,
);

testcase!(
    test_partial_as_callable_arg_mismatch_strict,
    TestEnv::new().enable_strict_partial_subtyping(),
    r#"
from functools import partial
from typing import Callable
def fn(a: int, b: str, c: bytes) -> int: ...
def callback1(fn: Callable[[str, bytes], int]) -> None: ...
def callback2(fn: Callable[[str, int], int]) -> None: ...
callback1(partial(fn, 1))
callback2(partial(fn, 1))  # E: Argument `(b: str, c: bytes) -> int` is not assignable to parameter `fn` with type `(str, int) -> int` in function `callback2`
"#,
);

// A residual returned as a declared `Callable[...]` (the IG coercer pattern). Default mode is
// gradual on parameters, so no mismatch is reported; strict flags it.
testcase!(
    test_partial_residual_return_gradual,
    TestEnv::new(),
    r#"
from functools import partial
from typing import Callable
def list_coercer(input_value: list[object], *, coercer: Callable[[object], object]) -> list[object]:
    return [coercer(x) for x in input_value]
def get_coercer(inner: Callable[[object], object]) -> Callable[[object], object]:
    return partial(list_coercer, coercer=inner)
"#,
);

testcase!(
    test_partial_residual_return_strict,
    TestEnv::new().enable_strict_partial_subtyping(),
    r#"
from functools import partial
from typing import Callable
def list_coercer(input_value: list[object], *, coercer: Callable[[object], object]) -> list[object]:
    return [coercer(x) for x in input_value]
def get_coercer(inner: Callable[[object], object]) -> Callable[[object], object]:
    return partial(list_coercer, coercer=inner)  # E: Returned type `(input_value: list[object], *, coercer: (object) -> object = ...) -> list[object]` is not assignable to declared return type `(object) -> object`
"#,
);

// Arity: a residual with fewer params than the callable target expects. Default (stub) mode is
// gradual, so it's not caught; strict catches it.
testcase!(
    test_partial_residual_arity_gradual,
    TestEnv::new(),
    r#"
from functools import partial
from typing import Callable
def store(inputs_key: str, precompile_key: str, a: int, b: int, c: int, d: int) -> None: ...
def add_saver(fn: Callable[[int, int, int, int, Callable[[], int]], None]) -> None: ...
add_saver(partial(store, "ik", "pk"))
"#,
);

testcase!(
    test_partial_residual_arity_strict,
    TestEnv::new().enable_strict_partial_subtyping(),
    r#"
from functools import partial
from typing import Callable
def store(inputs_key: str, precompile_key: str, a: int, b: int, c: int, d: int) -> None: ...
def add_saver(fn: Callable[[int, int, int, int, Callable[[], int]], None]) -> None: ...
add_saver(partial(store, "ik", "pk"))  # E: Argument `(a: int, b: int, c: int, d: int) -> None` is not assignable to parameter `fn` with type `(int, int, int, int, () -> int) -> None` in function `add_saver`
"#,
);

// Default mode matches mypy. Bound args and direct residual calls are checked and the residual
// signature is revealed, while the `Callable`-target assignment stays gradual (strict flags it).
testcase!(
    test_partial_default_mode_matches_mypy,
    TestEnv::new(),
    r#"
from typing import reveal_type, Callable
from functools import partial
def foo(a: int, b: int, c: str) -> str: return c
partial(foo, "a")  # E: Argument `Literal['a']` is not assignable to parameter `a` with type `int` in function `foo`
p = partial(foo, 1)
reveal_type(p)  # E: revealed type: (b: int, c: str) -> str
p(1, 3)  # E: Argument `Literal[3]` is not assignable to parameter `c` with type `str`
c: Callable[[int], str] = partial(foo, 1, 2)
"#,
);

// Strict mode checks the same bound args PLUS the residual as a subtype. It flags the
// `Callable`-target param and arity mismatches that default mode leaves gradual.
testcase!(
    test_partial_strict_mode_checks_residual,
    TestEnv::new().enable_strict_partial_subtyping(),
    r#"
from typing import reveal_type, Callable
from functools import partial
def foo(a: int, b: int, c: str) -> str: return c
partial(foo, "a")  # E: Argument `Literal['a']` is not assignable to parameter `a` with type `int` in function `foo`
p = partial(foo, 1)
reveal_type(p)  # E: revealed type: (b: int, c: str) -> str
p(1, 3)  # E: Argument `Literal[3]` is not assignable to parameter `c` with type `str`
c: Callable[[int], str] = partial(foo, 1, 2)  # E: `(c: str) -> str` is not assignable to `(int) -> str`
"#,
);

functools_testcase!(
    test_partial_class_object_arg_check,
    r#"
from typing import reveal_type
from functools import partial
class A:
    def __init__(self, var: int, b: int, c: int) -> None: ...
p = partial(A, 1)
reveal_type(p)  # E: revealed type: (b: int, c: int) -> A
p(1, "no")  # E: Argument `Literal['no']` is not assignable to parameter `c` with type `int`
q: partial[A] = partial(A, 1)
"#,
);

// Classes extending `Any` are gradual, so their constructors can be
// partially-applied with any arguments
functools_testcase!(
    test_partial_class_object_any_base_gradual,
    r#"
from typing import Any
from functools import partial
class C(Any):
    field: int
partial(C, field=1, other="x")
"#,
);

// A bare abstract class is flagged at partial construction, where the problem originates; a
// `type[A]` value can still be a concrete subclass, so it is not flagged.
functools_testcase!(
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
    partial_cls = partial(A)  # E: Cannot instantiate `A` because the following members are abstract: `method`
    partial_cls()
"#,
);

// A bare protocol is flagged at partial construction with the protocol-specific message, matching
// a direct `P()` call; a `type[P]` value can still be a concrete subclass, so it is not flagged.
functools_testcase!(
    test_partial_protocol_class,
    r#"
from functools import partial
from typing import Protocol
class P(Protocol):
    def method(self) -> None: ...
def f1(cls: type[P]) -> None:
    cls()
    partial_cls = partial(cls)
    partial_cls()
def f2() -> None:
    P()  # E: Cannot instantiate `P` because it is a protocol
    partial_cls = partial(P)  # E: Cannot instantiate `P` because it is a protocol
    partial_cls()
"#,
);

// ===== `partial[ret]` identity via subtyping =====

functools_testcase!(
    test_partial_assignable_to_partial_type,
    r#"
import functools
def f(a: int, b: str) -> int: ...
p: functools.partial[int] = functools.partial(f, 1)
"#,
);

functools_testcase!(
    test_partial_plain_callable_not_partial_type,
    r#"
import functools
from typing import Callable
def take(p: functools.partial[str]) -> None: ...
def main(c: Callable[[int], str]) -> None:
    take(c)  # E: Argument `(int) -> str` is not assignable to parameter `p` with type `partial[str]` in function `take`
"#,
);

// Cross-module: the wrapped function is imported, so the residual is synthesized across the
// import boundary.
testcase!(
    test_partial_cross_module,
    TestEnv::one("helper", "def make(a: int, b: str) -> bytes: return b''")
        .enable_strict_partial_subtyping(),
    r#"
from typing import reveal_type
import functools
from helper import make
p = functools.partial(make, 1)
reveal_type(p)  # E: revealed type: (b: str) -> bytes
p(2)  # E: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
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
