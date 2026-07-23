/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_generic_call_happy_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: T) -> tuple[S, T]: ...
force_error(f(1, "foo"))  # E: Argument `tuple[int, str]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_simple,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S) -> tuple[S, T]: ...
force_error(f(1))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_union_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: list[T] | None) -> tuple[S, T]: ...
force_error(f(1, None))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_self_type_subst,
    r#"
from typing import assert_type, Self
class A:
    def __new__(cls) -> Self: ...
class B[T](A): ...
class C[T]: ...
assert_type(A.__new__(A), A)
assert_type(A.__new__(B[int]), B[int])
assert_type(A.__new__(C[int]), C[int]) # E: `C[int]` is not assignable to upper bound `A` of type variable `Self@A`

o = A()
assert_type(o.__new__(A), A)
assert_type(o.__new__(B[int]), B[int])
assert_type(o.__new__(C[int]), C[int]) # E: `C[int]` is not assignable to upper bound `A` of type variable `Self@A`
    "#,
);

testcase!(
    test_self_type_subst_overloaded_dunder_new,
    r#"
from typing import Self, assert_type, overload
class C:
    @overload
    def __new__(cls, x: int) -> Self: ...
    @overload
    def __new__(cls, x: str) -> Self: ...
    def __new__(cls, x: int | str) -> Self:
        return super().__new__(cls)

assert_type(C.__new__(C, 0), C)
assert_type(C.__new__(C, ""), C)
    "#,
);

testcase!(
    test_self_type_subst_use_receiver,
    r#"
from typing import assert_type, Self
class A[T]:
    def __new__(cls: type[Self], x: T) -> Self: ...
# A[int] is a generic alias, which doesn't resolve to custom __new__
o = A[int].__new__(A[str], "foo") # E: Missing positional argument `args` in function `types.GenericAlias.__new__` # E: `A[str]` is not assignable to upper bound `GenericAlias` of type variable `Self@GenericAlias` # E: Argument `Literal['foo']` is not assignable to parameter `origin` with type `type[Any]` in function `types.GenericAlias.__new__`
    "#,
);

testcase!(
    test_deprecated_call,
    r#"
from warnings import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...
old_function()  # E: `old_function` is deprecated
    "#,
);

fn test_env_3_12() -> TestEnv {
    TestEnv::new_with_version(PythonVersion {
        major: 3,
        minor: 12,
        micro: 0,
    })
}

testcase!(
    test_deprecated_call_3_12,
    test_env_3_12(),
    r#"
from typing_extensions import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...
old_function()  # E: `old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_function_reference,
    r#"
from typing import Callable
from warnings import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...

def take_callable(f: Callable) -> None: ...
take_callable(old_function)  # E: `old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_method_call,
    r#"
from warnings import deprecated
class C:
    @deprecated("function is deprecated")
    def old_function(self) -> None: ...

c = C()
c.old_function()  # E: `C.old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_overloaded_call,
    r#"
from typing import overload
from warnings import deprecated

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
@deprecated("DEPRECATED")
def f(x: int | str) -> int | str:
    return x

f(0)  # E: `f` is deprecated
    "#,
);

fn test_env_string_as_iterable() -> TestEnv {
    TestEnv::new().enable_string_as_iterable_warning()
}

testcase!(
    test_string_as_iterable_warning,
    test_env_string_as_iterable(),
    r#"
from typing import Iterable, Sequence

def takes_iter(xs: Iterable[str]) -> None: ...
def takes_seq(xs: Sequence[str]) -> None: ...
def takes_iter_or_str(xs: Iterable[str] | str) -> None: ...

s: str = "hello"
takes_iter(s)  # E: Passing `str` to `Iterable[str]` treats the string as an iterable of characters
takes_seq(s)  # E: Passing `str` to `Sequence[str]` treats the string as an iterable of characters
takes_iter_or_str(s)
takes_iter(["hello"])

x: Iterable[str] = s  # E: Passing `str` to `Iterable[str]` treats the string as an iterable of characters
y: Sequence[str] = s  # E: Passing `str` to `Sequence[str]` treats the string as an iterable of characters

takes_iter("hello")  # E: Passing `str` to `Iterable[str]` treats the string as an iterable of characters
z: Iterable[str] = "hello"  # E: Passing `str` to `Iterable[str]` treats the string as an iterable of characters
    "#,
);

testcase!(
    test_string_as_iterable_warning_does_not_break_overload_matching,
    test_env_string_as_iterable(),
    r#"
from traceback import format_exception

def f(exc: BaseException) -> None:
    "".join(format_exception(exc))

s: str = "hello"
"".join(s)  # E: Passing `str` to `Iterable[str]` treats the string as an iterable of characters
    "#,
);

testcase!(
    test_deprecated_overloaded_signature,
    r#"
from typing import overload
from warnings import deprecated

@deprecated("DEPRECATED")
@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

f(0)  # E: Call to deprecated overload `f`
f("foo") # No error
    "#,
);

testcase!(
    test_deprecated_overloaded_signature_no_impl,
    r#"
from typing import overload
from warnings import deprecated

@deprecated("DEPRECATED")
@overload
def f(x: int) -> int: ...  # E: Overloaded function must have an implementation
@overload
def f(x: str) -> str: ...

f(0)  # E: Call to deprecated overload `f`
f("foo") # No error
    "#,
);

testcase!(
    test_nondeprecated_overload_shutil,
    r#"
import shutil
shutil.rmtree("/tmp")
    "#,
);

testcase!(
    test_deprecated_message,
    r#"
from warnings import deprecated
@deprecated("I am a special super-important message about the extended warranty on your car")
def f(): ...

f()  # E: I am a special super-important message about the extended warranty on your car
    "#,
);

testcase!(
    test_deprecated_fqn,
    r#"
import warnings
@warnings.deprecated("Deprecated")
def f(): ...
f()  # E: Deprecated
    "#,
);

testcase!(
    test_reduce_call,
    r#"
from functools import reduce
reduce(max, [1,2])
    "#,
);

testcase!(
    test_call_arg_lambda_contextual_typing,
    r#"
from typing import Callable

def takes(cb: Callable[[int], int]) -> None: ...

# This only errors because we're able to pass down the `int` hint through contextual typing.
takes(lambda x: x + "")  # E:  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`
    "#,
);

testcase!(
    test_union_with_type,
    r#"
from typing import assert_type
class A:
    pass
def identity[T](x: T) -> T:
    return x
def f(condition: bool):
    if condition:
        g = type
    else:
        g = identity
    assert_type(g(A()), type[A] | A)
    "#,
);

testcase!(
    test_generic_function_subscript,
    r#"
def func[T](x: T) -> T:
    return x

func[int](100)  # E: `func` is not subscriptable
    "#,
);

testcase!(
    test_any_constructor,
    r#"
from typing import Any
Any()  # E: `Any` cannot be instantiated
    "#,
);

testcase!(
    test_object_new_explicit_call,
    r#"
from typing import assert_type

class A: pass
class B(A): pass

# Direct object.__new__ calls should return the argument class type
x1 = object.__new__(A)
assert_type(x1, A)

x2 = object.__new__(B)
assert_type(x2, B)

# Works with builtin classes too
x3 = object.__new__(int)
assert_type(x3, int)

# Works with `type` annotations too
def f(cls: type[A]):
    x4 = object.__new__(cls)
    assert_type(x4, A)
    "#,
);

testcase!(
    test_object_new_with_generics,
    r#"
from typing import assert_type

class Container[T]: pass

# object.__new__ with generic class should preserve type params
x = object.__new__(Container[int])
assert_type(x, Container[int])
    "#,
);

testcase!(
    test_custom_new_unaffected,
    r#"
from typing import Self, assert_type

class A[T]:
    def __new__(cls: type[Self], x: T) -> Self: ...

# A[int] is a generic alias, which doesn't resolve to custom __new__
o = A[int].__new__(A[int], 42) # E: Missing positional argument `args` in function `types.GenericAlias.__new__` # E: `A[int]` is not assignable to upper bound `GenericAlias` of type variable `Self@GenericAlias` # E: Argument `Literal[42]` is not assignable to parameter `origin` with type `type[Any]` in function `types.GenericAlias.__new__`
assert_type(o, A[int])

# Receiver type binding is preserved
class B:
    def __new__(cls) -> Self: ...

b = B.__new__(B)
assert_type(b, B)
    "#,
);

testcase!(
    test_inherit_custom_new,
    r#"
from typing import assert_type, Self
class A:
    def __new__(cls) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A().__new__(B), B)
assert_type(A.__new__(B), B)
    "#,
);

testcase!(
    test_inherit_generic_custom_new,
    r#"
from typing import assert_type, Self
class A:
    def __new__[T](cls, x: T, y: T) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A.__new__(B, 0, 0), B)
    "#,
);

testcase!(
    test_inherit_overloaded_custom_new,
    r#"
from typing import assert_type, overload, Self
class A:
    @overload
    def __new__(cls) -> Self: ...
    @overload
    def __new__(cls, x) -> Self: ...
    def __new__(cls, x=None) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A.__new__(B), B)
assert_type(A.__new__(B, 0), B)
    "#,
);

// Minimized from https://github.com/PrefectHQ/prefect/blob/3e80a036349748edfac2ccb5609f65b7f91e85d8/src/prefect/runtime/flow_run.py#L218.
testcase!(
    test_complicated_paramspec_forwarding,
    r#"
from collections.abc import Awaitable
from typing import assert_type, Callable

type _SyncOrAsyncCallable[**P, T] = Callable[P, T | Awaitable[T]]

class Flow: ...

class Call[T]:
    def __call__(self) -> T | Awaitable[T]: ...
    def result(self) -> T: ...

def create_call[**P, T](
    fn: _SyncOrAsyncCallable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Call[T]: ...

def call_soon_in_loop_thread[T](
    call: _SyncOrAsyncCallable[[], T] | Call[T],
) -> Call[T]: ...

async def _get_flow_from_run(flow_run_id: str) -> Flow: ...

def get_flow_version(run_id: str | None) -> str | None:
    flow = call_soon_in_loop_thread(
        create_call(_get_flow_from_run, run_id)  # E: `str | None` is not assignable to parameter `flow_run_id`
    ).result()
    assert_type(flow, Flow)
    "#,
);

testcase!(
    test_call_not_implemented_constant,
    r#"
# NotImplemented is a singleton constant, not a callable class.
# Using NotImplemented() is always a mistake; they mean NotImplementedError().
def broken():
    raise NotImplemented()  # E: `NotImplemented` is not callable. Did you mean `NotImplementedError`?

def also_broken():
    raise NotImplemented("not yet done")  # E: `NotImplemented` is not callable. Did you mean `NotImplementedError`?
"#,
);

testcase!(
    test_call_not_implemented_in_union,
    r#"
def f(condition: bool):
    def g(): ...
    if condition:
        x = g
    else:
        x = NotImplemented
    x()  # E: `NotImplemented` is not callable. Did you mean `NotImplementedError`?
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2914
testcase!(
    test_non_callable_bool_attribute,
    r#"
class BadBool:
    __bool__: int = 3

assert BadBool()  # E: `__bool__` attribute of `BadBool` has type `int`, which is not callable
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/3060
testcase!(
    test_setdefault_then_index,
    r#"
def parse_groups(entries: list[tuple[str, str]]) -> None:
    groups = {}
    for group, host in entries:
        groups.setdefault(group, {})
        groups[group][host] = True
"#,
);

testcase!(
    test_call_instance_with_non_callable_dunder_call,
    r#"
class Uncallable:
    __call__ = 42

obj = Uncallable()
obj()  # E: Expected a callable, got `Uncallable`
"#,
);

// Verify **kwargs unpacking correctly suppresses missing-argument errors.
testcase!(
    test_kwargs_unpacking_provides_required_args,
    r#"
class Config:
    def __init__(self, name: str, value: int) -> None: ...

data = {"name": "test", "value": 42}
Config(**data)

def make(**kwargs) -> Config:
    return Config(**kwargs)
    "#,
);

// Verify object.__init__(self) is accepted when self is the only argument.
testcase!(
    test_object_init_explicit_self,
    r#"
class MyClass:
    def __init__(self) -> None:
        object.__init__(self)

class MyClass2:
    def __new__(cls) -> "MyClass2":
        return object.__new__(cls)
    "#,
);

// Verify explicit kwarg + **dict_mapping doesn't falsely report conflicts.
testcase!(
    test_explicit_kwarg_with_mapping_kwargs,
    r#"
class Config:
    def __init__(self, name: str, value: int, **kwargs) -> None: ...

extra = {"debug": True}
Config(name="test", value=42, **extra)
    "#,
);

testcase!(
    test_bad_argument_type_none_hint,
    r#"
def takes_str(x: str) -> None: ...

maybe: str | None = "hello"
takes_str(maybe)  # E: Consider narrowing the value with an `is not None` check  # !E: changing the declared type

takes_str(None)  # E:  # !E: `is not None` check  # !E: changing the declared type

if maybe is not None:
    takes_str(maybe)  # OK — narrowed
    "#,
);

testcase!(
    test_bad_assignment_none_hint,
    r#"
maybe: str | None = "hello"
x: int | str = maybe  # E: Consider narrowing the value with an `is not None` check or changing the declared type to `int | str | None`
    "#,
);

testcase!(
    test_bad_return_none_hint,
    r#"
def foo(x: bool) -> str:
    if x:
        y = "hello"
    else:
        y = None
    return y  # E: Consider narrowing the value with an `is not None` check or changing the declared type to `str | None`
    "#,
);

testcase!(
    test_implicit_return_no_none_hint,
    r#"
def f() -> str:  # E:  # !E: does not allow `None`
    pass
def g(x: str) -> str:  # E:  # !E: does not allow `None`
    if x:
        return x
    "#,
);

testcase!(
    test_bad_default_none_hint,
    r#"
def default() -> int | None: ...
def f(x: int = default()):  # E: Consider changing the declared type to `int | None`  # !E: `is not None` check
    pass
    "#,
);

testcase!(
    test_bare_none_hint,
    r#"
x: str = None  # E: Consider changing the declared type to `str | None`  # !E: `is not None` check
    "#,
);

testcase!(
    test_attribute_assignment_none_hint,
    r#"
class A:
    def __init__(self):
        self.x = 42

def f(a: A, x: int | None):
    a.x = x  # E: Consider narrowing the value with an `is not None` check  # !E: changing the declared type

def g(a: A):
    a.x = None  # E:  # !E: `is not None` check  # !E: changing the declared type
    "#,
);

testcase!(
    test_return_hint_not_used_if_detrimental,
    r#"
from collections.abc import Callable
from typing import reveal_type

def first[T](items: list[T], matcher: Callable[[T], bool]) -> T | None: ...
def foo(items: list[int]) -> int | None:
    return first(items, lambda i: reveal_type(i) == 3)  # E: revealed type: int
    "#,
);

testcase!(
    test_uses_return_hint_even_if_some_arg_error,
    r#"
from collections.abc import Iterable
from typing import Any

def collect[T](xs: Iterable[T], unrelated: Any) -> list[T]: ...

def f() -> list[object]:
    return collect(["x"], 1 + "oops")  # E: `+` is not supported between `Literal[1]` and `Literal['oops']`
    "#,
);
