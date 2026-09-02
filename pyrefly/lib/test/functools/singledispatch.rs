/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.singledispatch` test suite. Divergences are `bug=`-marked; `# WANT:` records the
//! correct target (flip by dropping `bug=` and turning `# WANT: X` into `# E: X`).

use crate::functools_testcase;

// The dispatch (first) argument is not checked against the fallback's first parameter: `1` has no
// registered impl and is not a subtype of `A`, but at runtime it falls through to the fallback.
functools_testcase!(
    test_singledispatch_unregistered_dispatch_arg_ok,
    r#"
from functools import singledispatch

class A: pass
class B(A): pass

@singledispatch
def fun(arg: A) -> None:
    pass
@fun.register
def fun_b(arg: B) -> None:
    pass

fun(1)
"#,
);

functools_testcase!(
    test_singledispatch_multiple_underscore_impls_ok,
    r#"
from functools import singledispatch

@singledispatch
def fun(arg) -> None:
    pass
@fun.register
def _(arg: str) -> None:
    pass
@fun.register
def _(arg: int) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_non_dispatch_arg_checked,
    r#"
from functools import singledispatch

class A: pass
class B(A): pass

@singledispatch
def f(arg: A, arg2: str) -> None:
    pass

@f.register
def g(arg: B, arg2: str) -> None:
    pass

f(A(), 'a')
f(A(), 5)  # E: Argument `Literal[5]` is not assignable to parameter `arg2` with type `str`

f(B(), 'a')
f(B(), 1)  # E: Argument `Literal[1]` is not assignable to parameter `arg2` with type `str`
"#,
);

functools_testcase!(
    bug = "pyrefly flags `x` as uninitialized at the call site; the union member is a valid annotated declaration and the dispatched call should be accepted",
    test_singledispatch_union_dispatch_arg,
    r#"
from functools import singledispatch
from typing import Union

class A: pass
class B(A): pass
class C: pass

@singledispatch
def f(arg: Union[A, C]) -> None:
    pass

@f.register
def g(arg: B) -> None:
    pass

@f.register
def h(arg: C) -> None:
    pass

x: Union[B, C]
# WANT: no error (`x: Union[B, C]` is a declared, usable variable)
f(x)  # E: `x` is uninitialized
"#,
);

functools_testcase!(
    test_singledispatch_union_arg_partly_unregistered,
    r#"
from functools import singledispatch
from typing import Union

class A: pass
class B(A): pass
class C: pass

@singledispatch
def f(arg: Union[A, C]) -> None:
    pass

@f.register
def g(arg: B) -> None:
    pass

@f.register
def h(arg: C) -> None:
    pass

def use(x: Union[B, C, int]) -> None:
    f(x)
"#,
);

functools_testcase!(
    test_singledispatch_abc_dispatch_type_ok,
    r#"
from functools import singledispatch
from collections.abc import Mapping

@singledispatch
def f(arg) -> None:
    pass

@f.register
def g(arg: Mapping) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_dispatcher_bad_signatures,
    r#"
from functools import singledispatch

@singledispatch
def f() -> None: # E: Singledispatch function requires at least one parameter
    pass

@singledispatch
def g(**kwargs) -> None: # E: First parameter of a singledispatch function must be positional
    pass

@singledispatch
def h(*, x) -> None: # E: First parameter of a singledispatch function must be positional
    pass

@singledispatch
def i(*, x=1) -> None: # E: First parameter of a singledispatch function must be positional
    pass
"#,
);

functools_testcase!(
    test_singledispatch_multiple_dispatchers_intermixed,
    r#"
from typing import reveal_type, assert_type
from functools import singledispatch

class A: pass
class B(A): pass
class C: pass

@singledispatch
def f(arg: A) -> None:
    pass

@singledispatch
def h(arg: C) -> None:
    pass

@f.register
def g(arg: B) -> None:
    pass
"#,
);

// Regression: https://github.com/facebook/pyrefly/issues/1006
functools_testcase!(
    test_singledispatch_dispatcher_with_default_arg,
    r#"
from typing import reveal_type
from functools import singledispatch
from typing import Any
@singledispatch
def fun(arg: Any, verbose: bool = False) -> None: ...
@fun.register
def _(arg: int, verbose: bool = False) -> None: ...
@fun.register
def _(arg: str, verbose: bool = False) -> None: ...
fun(1)
fun("a")
fun(1.0)
reveal_type(fun)  # E: revealed type: _SingleDispatchCallable[None]
reveal_type(fun(1))  # E: revealed type: None
"#,
);

// A raising fallback with no return annotation infers `Never`, widened to gradual `Any` so the
// dispatcher still accepts registered implementations.
functools_testcase!(
    test_singledispatch_raising_fallback_registers_ok,
    r#"
from functools import singledispatch
@singledispatch
def fun(arg):
    raise NotImplementedError
@fun.register
def _(arg: int) -> int: return -arg
"#,
);

functools_testcase!(
    test_singledispatch_raising_fallback_element_is_gradual,
    r#"
from functools import singledispatch
from typing import reveal_type
@singledispatch
def fun(arg):
    raise NotImplementedError
reveal_type(fun)  # E: revealed type: _SingleDispatchCallable[Unknown]
"#,
);

// Calling a raising-fallback dispatcher yields gradual `Any`, not `Never`: at runtime the call
// dispatches to a registered impl and returns a real value, so `Never` would be unsound.
functools_testcase!(
    test_singledispatch_raising_fallback_call_is_gradual,
    r#"
from functools import singledispatch
from typing import reveal_type
@singledispatch
def fun(arg):
    raise NotImplementedError
@fun.register
def _(arg: int) -> int: return -arg
reveal_type(fun(1))  # E: revealed type: Unknown
"#,
);

functools_testcase!(
    test_singledispatch_raising_fallback_multiple_registrations_ok,
    r#"
from functools import singledispatch
@singledispatch
def fun(arg):
    raise NotImplementedError
@fun.register
def _(arg: int) -> int: return -arg
@fun.register
def _(arg: str) -> str: return arg
@fun.register
def _(arg: bytes) -> bytes: return arg
"#,
);

// Raising a concrete exception, not just `NotImplementedError`, is also a `Never` return.
functools_testcase!(
    test_singledispatch_raising_fallback_concrete_exception_ok,
    r#"
from functools import singledispatch
@singledispatch
def fun(arg):
    raise ValueError("no default")
@fun.register
def _(arg: int) -> int: return -arg
"#,
);

// An annotated fallback's element type is its declared return type.
functools_testcase!(
    test_singledispatch_annotated_fallback_keeps_return_type,
    r#"
from functools import singledispatch
from typing import reveal_type
@singledispatch
def fun(arg) -> int:
    return 0
reveal_type(fun)  # E: revealed type: _SingleDispatchCallable[int]
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_dispatched_call_checks_fallback_sig,
    r#"
from functools import singledispatch
from typing import reveal_type
@singledispatch
def f(arg: int) -> str:
    return str(arg)
reveal_type(f(1))  # E: revealed type: str
f("not an int")
f()  # E: Missing argument `arg`
"#,
);

// Dispatch happens at runtime on the first argument, which may match a registered impl whose type
// is not a subtype of the fallback's first parameter, so that argument is not checked against it.
functools_testcase!(
    test_singledispatch_call_registered_non_subtype_arg,
    r#"
from functools import singledispatch
@singledispatch
def f(arg: int) -> str:
    return str(arg)
@f.register  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
def _(arg: str) -> str:
    return arg
f("hello")
"#,
);

// A `*args` fallback dispatches on the first vararg, so widening treats the vararg element as the
// dispatch position and a call routed to a registered impl of another type is accepted.
functools_testcase!(
    test_singledispatch_varargs_fallback_call,
    r#"
from functools import singledispatch
@singledispatch
def f(*args: int) -> str:
    return "fallback"
@f.register  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
def _(arg: str) -> str:
    return arg
f("hello")
"#,
);

// A `@singledispatch` implementation may carry `@overload` declarations describing the registered
// dispatch variants, whose signatures differ from the fallback; that is not an inconsistent overload.
functools_testcase!(
    test_singledispatch_overloaded_dispatcher_no_inconsistency,
    r#"
from functools import singledispatch
from typing import overload, Callable, TypeVar
T = TypeVar("T")
@overload
def impl(qualname: str, func: Callable[..., T] | None = None) -> object: ...
@overload
def impl(lib: int, name: str, key: str = "") -> object: ...
@singledispatch
def impl(qualname: str, func: Callable[..., T] | None = None) -> object:
    return None
"#,
);
// A generic fallback keeps its type params, so a dispatched call binds the type variable from the
// argument instead of collapsing the return to Unknown. Dispatch-param widening stays skipped for
// the type-variable parameter so the binding is not severed.
functools_testcase!(
    test_singledispatch_generic_fallback_binds_return,
    r#"
from functools import singledispatch
from typing import reveal_type, TypeVar
T = TypeVar("T")
@singledispatch
def f(arg: T) -> T:
    return arg
reveal_type(f(1))  # E: revealed type: int
"#,
);

// A generic dispatcher with a concrete dispatch parameter still widens that parameter, so a call
// with a registered (non-fallback) dispatch type is accepted while other type vars still bind.
functools_testcase!(
    test_singledispatch_generic_fallback_widens_concrete_dispatch_param,
    r#"
from functools import singledispatch
from typing import reveal_type, TypeVar
T = TypeVar("T")
class A: pass
class B(A): pass
@singledispatch
def f(arg: A, x: T) -> T:
    return x
@f.register
def _(arg: B, x: T) -> T:
    return x
reveal_type(f(B(), 1))  # E: revealed type: int
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_malformed_dispatcher,
    r#"
from functools import singledispatch
@singledispatch
def f() -> None: ...  # E: Singledispatch function requires at least one parameter
@singledispatch
def g(*, x: int) -> None: ...  # E: First parameter of a singledispatch function must be positional
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_dispatch_and_registry,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> None: ...
@f.register
def _(arg: int) -> None: ...
reveal_type(f.dispatch(int))  # E: revealed type: (...) -> None
reveal_type(f.registry)  # E: revealed type: MappingProxyType[Any, (...) -> None]
"#,
);

// An annotated `singledispatchmethod` fallback whose body only raises keeps its declared return:
// the annotation pins it, so `Never` does not leak into the revealed call type.
functools_testcase!(
    test_singledispatchmethod_raising_fallback_annotated_return,
    r#"
from functools import singledispatchmethod
from typing import reveal_type
class Negator:
    @singledispatchmethod
    def neg(self, arg) -> object:
        raise NotImplementedError
    @neg.register
    def _(self, arg: int) -> int: return -arg
    @neg.register
    def _(self, arg: bool) -> bool: return not arg
n = Negator()
reveal_type(n.neg(5))  # E: revealed type: object
"#,
);

// `singledispatchmethod` is not modeled (only the `singledispatch` function form), so an
// unannotated raising fallback collapses to `Never` and `.register` is spuriously rejected.
functools_testcase!(
    bug = "unannotated raising singledispatchmethod fallback poisons the element type to Never, so .register is spuriously rejected",
    test_singledispatchmethod_unannotated_raising_fallback_poisons,
    r#"
from functools import singledispatchmethod
class Negator:
    @singledispatchmethod
    def neg(self, arg):
        raise NotImplementedError
    @neg.register  # E: No matching overload found for function `functools.singledispatchmethod.register`
    def _(self, arg: int) -> int: return -arg
"#,
);
