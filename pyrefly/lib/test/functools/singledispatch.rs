/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.singledispatch` conformance: dispatcher definition and calling the dispatcher.
//!
//! pyrefly has no native singledispatch modeling (it relies on the typeshed `_SingleDispatchCallable`
//! stub, whose `__call__(*args, **kwargs) -> _T` erases argument info), so it gets the dispatcher's
//! return type right but does not validate dispatch/non-dispatch args or the dispatcher signature.
//! Divergences are `bug=`-marked; `# WANT:` records the correct target. To flip a test: drop
//! `bug=` and turn each `# WANT: X` into `# E: X` (or delete a now-spurious `# E:`).

use crate::functools_testcase;

functools_testcase!(
    bug = "calling a singledispatch function with an argument incompatible with the fallback type should error, but pyrefly is silent (stub-driven dispatch loses the param type)",
    test_singledispatch_call_arg_mismatches_fallback,
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

fun(1)  # WANT: Argument 1 to "fun" has incompatible type "int"; expected "A"
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
    bug = "pyrefly does not check non-dispatch arguments of singledispatch calls; type mismatches on arg2 go undetected",
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
f(A(), 5)  # WANT: Argument 2 to "f" has incompatible type "int"; expected "str"

f(B(), 'a')
f(B(), 1)  # WANT: Argument 2 to "f" has incompatible type "int"; expected "str"
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
    bug = "pyrefly does not validate singledispatch dispatch argument types: `B | C | int` is not assignable to the dispatcher's declared `A | C` (the `int` part has no registered impl), but pyrefly emits no error",
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
    # WANT: Argument 1 to "f" has incompatible type "B | C | int"; expected "A | C"
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
    bug = "pyrefly does not validate singledispatch function signatures; all four malformed-dispatcher cases are missed (false negatives)",
    test_singledispatch_dispatcher_bad_signatures,
    r#"
from functools import singledispatch

@singledispatch
def f() -> None: # WANT: Singledispatch function requires at least one argument
    pass

@singledispatch
def g(**kwargs) -> None: # WANT: First argument to singledispatch function must be a positional argument
    pass

@singledispatch
def h(*, x) -> None: # WANT: First argument to singledispatch function must be a positional argument
    pass

@singledispatch
def i(*, x=1) -> None: # WANT: First argument to singledispatch function must be a positional argument
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

// Edge case
functools_testcase!(
    bug = "dispatched singledispatch calls are not checked against the fallback signature: bad arg types and missing args go unreported",
    test_singledispatch_dispatched_call_checks_fallback_sig,
    r#"
from functools import singledispatch
from typing import reveal_type
@singledispatch
def f(arg: int) -> str:
    return str(arg)
reveal_type(f(1))  # E: revealed type: str
# WANT: arg-type error (str not assignable to int)
f("not an int")
# WANT: missing-argument error (arg)
f()
"#,
);

// Edge case
functools_testcase!(
    bug = "singledispatch with no positional args or a keyword-only first arg should be rejected; pyrefly emits no error",
    test_singledispatch_malformed_dispatcher,
    r#"
from functools import singledispatch
@singledispatch
# WANT: error: Singledispatch function requires at least one argument
def f() -> None: ...
@singledispatch
# WANT: error: First argument to singledispatch function cannot be keyword-only
def g(*, x: int) -> None: ...
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
