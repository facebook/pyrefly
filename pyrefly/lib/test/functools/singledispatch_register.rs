/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.singledispatch` `.register` behavior: dispatch-type vs fallback subtype checks,
//! register-type vs impl-signature checks, explicit/functional registration, and calling a
//! registered implementation directly. pyrefly is stub-driven, so most register-time checks are
//! missed (false negatives); divergences are `bug=`-marked with the target as `# WANT: ...`.

use crate::functools_testcase;

functools_testcase!(
    bug = "singledispatch fallback shadowed by a registered impl with the same dispatch type as the fallback's first arg should be flagged as never-used; pyrefly is stub-driven and emits nothing",
    test_singledispatch_register_same_type_as_fallback_shadows,
    r#"
from functools import singledispatch

# WANT: singledispatch implementation 1 will never be used: implementation 2's dispatch type is the same
@singledispatch
def f(arg: int) -> None:
    pass

@f.register
def g(arg: int) -> None:
    pass
"#,
);

functools_testcase!(
    bug = "register(str) should error when the impl's first param is typed int, but pyrefly does not check dispatch-type vs signature",
    test_singledispatch_register_type_mismatches_signature,
    r#"
from functools import singledispatch

@singledispatch
def f(arg) -> None:
    pass

# WANT: Argument to register "str" is incompatible with type "int" in function signature
@f.register(str)
def g(arg: int) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_register_explicit_type_not_subtype,
    r#"
from functools import singledispatch

@singledispatch
def f(arg: int) -> None:
    pass

@f.register(str)  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
def g(arg) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_varargs_fallback_register_not_subtype,
    r#"
from functools import singledispatch

@singledispatch
def f(*args: int) -> None:
    pass

@f.register(str)  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
def g(arg) -> None:
    pass
"#,
);

// A `.register(C)` argument is a value, not a type form, so a name that collides with a special
// form (here `typing.Union`) must not be misread as that form and flagged `invalid-annotation`.
functools_testcase!(
    test_singledispatch_register_special_form_named_arg_no_error,
    r#"
from functools import singledispatch
from typing import Union
@singledispatch
def f(op: object) -> None: ...
@f.register(Union)
def _(op) -> None: ...
"#,
);

functools_testcase!(
    test_singledispatch_custom_class_passed_as_type_to_register,
    r#"
from functools import singledispatch
class A: pass

@singledispatch
def f(arg: int) -> None:
    pass
@f.register(A)  # E: Dispatch type `A` is not a subtype of fallback first argument type `int`
def g(arg) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_dispatch_type_is_not_asubtype_of_fallback_first_argument,
    r#"
from functools import singledispatch

class A: pass
class B(A): pass
class C: pass

@singledispatch
def f(arg: A) -> None:
    pass

@f.register
def g(arg: B) -> None:
    pass

@f.register  # E: Dispatch type `C` is not a subtype of fallback first argument type `A`
def h(arg: C) -> None:
    pass
"#,
);

functools_testcase!(
    test_singledispatch_register_class_with_any_ctor_not_subtype,
    r#"
from functools import singledispatch
from typing import Any

class Base: pass
class ConstExpr:
    def __init__(self, **kwargs: Any) -> None: pass

@singledispatch
def f(arg: Base) -> ConstExpr:  # E: Function declared to return `ConstExpr` but is missing an explicit `return`
    pass

@f.register(ConstExpr)  # E: Dispatch type `ConstExpr` is not a subtype of fallback first argument type `Base`
def g(arg: ConstExpr) -> ConstExpr:  # E: Function declared to return `ConstExpr` but is missing an explicit `return`
    pass
"#,
);

functools_testcase!(
    bug = "pyrefly flags `x` as uninitialized; the bare `x: Missing` annotation is a declared local and the registered impl using `f` before all registrations are defined is fine (no error)",
    test_singledispatch_dispatch_call_from_registered_impl,
    r#"
from functools import singledispatch
from typing import Union

class Node: pass
class Module(Node): pass
class Missing: pass

@singledispatch
def f(a: Union[Node, Missing]) -> None:
    pass

@f.register
def g(a: Module) -> None:
    x: Missing
    # WANT: no error (`x: Missing` is a declared local; calling dispatch `f` from a registered impl before later registrations are defined is valid)
    f(x)  # E: `x` is uninitialized

@f.register
def h(a: Missing) -> None:
    pass
"#,
);

// Bare `@f.register` returns the impl's own type, so direct calls to the registered function are
// argument-checked.
functools_testcase!(
    test_singledispatch_registered_impl_bare_direct_call_arg_check,
    r#"
from functools import singledispatch

@singledispatch
def f(arg, arg2: str) -> bool:
    return False

@f.register
def g(arg: int, arg2: str) -> bool:
    return True

g('a', 'a')  # E: Argument `Literal['a']` is not assignable to parameter `arg` with type `int` in function `g`
g(1, 1)  # E: Argument `Literal[1]` is not assignable to parameter `arg2` with type `str` in function `g`
g(1, 'a')
"#,
);

// Explicit `@f.register(C)` returns the impl's own type, so direct calls to the registered function
// are argument-checked (its unannotated first parameter is not).
functools_testcase!(
    test_singledispatch_registered_impl_explicit_direct_call_arg_check,
    r#"
from functools import singledispatch

@singledispatch
def f(arg, arg2: str) -> bool:
    return False

@f.register(str)
def h(arg, arg2: str) -> bool:
    return True

h(1, 'a')
h('a', 1)  # E: Argument `Literal[1]` is not assignable to parameter `arg2` with type `str` in function `h`
"#,
);

functools_testcase!(
    bug = "pyrefly does not emit the 'Need type annotation' error when singledispatch is given a non-callable argument",
    test_singledispatch_register_after_noncallable_arg,
    r#"
from typing import reveal_type, assert_type
import functools
# WANT: Need type annotation for "a"
a = functools.singledispatch('a')  # E: Argument `Literal['a']` is not assignable to parameter `func` with type `(...) -> @_` in function `functools.singledispatch`

@a.register(int)
def default(val) -> int:
    return 3
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_register_subtype_of_fallback,
    r#"
from functools import singledispatch
class A: pass
class B(A): pass
class C: pass
@singledispatch
def f(arg: A) -> None: ...
@f.register
def g(arg: B) -> None: ...
@f.register  # E: Dispatch type `C` is not a subtype of fallback first argument type `A`
def h(arg: C) -> None: ...
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_register_explicit_type_decorator,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return ""
@f.register(int)
def _(arg: int) -> str: return "int"
reveal_type(f(1))  # E: revealed type: str
"#,
);

// Edge case
functools_testcase!(
    test_singledispatch_register_functional_form,
    r#"
from typing import reveal_type, assert_type
from functools import singledispatch
def f_impl(arg: object) -> str: return ""
def int_impl(arg: int) -> str: return "int"
g = singledispatch(f_impl)
reveal_type(g)  # E: revealed type: _SingleDispatchCallable[str]
g.register(int, int_impl)
reveal_type(g(1))  # E: revealed type: str
"#,
);

functools_testcase!(
    test_singledispatch_register_bare_functional_preserves_signature,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> str: return "int"
g = f.register(impl)
reveal_type(g)  # E: revealed type: (arg: int) -> str
g("wrong")  # E: Argument `Literal['wrong']` is not assignable to parameter `arg` with type `int` in function `impl`
"#,
);

functools_testcase!(
    test_singledispatch_register_two_step_call_preserves_signature,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> str: return "int"
reveal_type(f.register(int)(impl))  # E: revealed type: (arg: int) -> str
deco = f.register(int)
registered = deco(impl)
reveal_type(registered)  # E: revealed type: (arg: int) -> str
registered("wrong")  # E: Argument `Literal['wrong']` is not assignable to parameter `arg` with type `int` in function `impl`
"#,
);

// Registration is not position-sensitive: `f.register(...)` returns the impl and never mutates the
// dispatcher, so `f(...)` stays fallback-typed above and below the register call.
functools_testcase!(
    test_singledispatch_register_not_position_sensitive,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> int: return arg
reveal_type(f(1))  # E: revealed type: str
r = f.register(int)(impl)
reveal_type(r)  # E: revealed type: (arg: int) -> int
reveal_type(f(1))  # E: revealed type: str
"#,
);

// An overloaded impl applied by call keeps its overloads (not the stub's erased `(...) -> _T`), so a
// call that matches no overload is still reported.
functools_testcase!(
    test_singledispatch_register_two_step_overloaded_impl,
    r#"
from functools import singledispatch
from typing import overload
@singledispatch
def f(arg: object) -> str: return "base"
@overload
def impl(arg: int) -> str: ...
@overload
def impl(arg: bytes) -> str: ...
def impl(arg: object) -> str: return "int"
registered = f.register(int)(impl)
registered("wrong")  # E: No matching overload found for function `impl`
"#,
);

// A functional or bare register call returns the impl itself, so calling the captured result yields
// the impl's return type rather than being misread as applying a register decorator.
functools_testcase!(
    test_singledispatch_register_functional_capture_call,
    r#"
from typing import reveal_type
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> str: return "int"
g = f.register(int, impl)
reveal_type(g(1))  # E: revealed type: str
h = f.register(impl)
reveal_type(h(1))  # E: revealed type: str
"#,
);

// A malformed application of the register decorator (wrong arity, or a non-callable argument) is
// checked normally rather than silently taking the argument's type.
functools_testcase!(
    test_singledispatch_register_two_step_call_malformed,
    r#"
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> str: return "int"
f.register(int)(impl, impl)  # E: Expected 1 positional argument, got 2
f.register(int)(42)  # E: Argument `Literal[42]` is not assignable to parameter with type `(...) -> str`
"#,
);

// The non-subtype dispatch class is checked once at the `.register(C)` call, not re-reported when the
// returned decorator is applied.
functools_testcase!(
    test_singledispatch_register_two_step_call_not_subtype,
    r#"
from functools import singledispatch
@singledispatch
def f(arg: int) -> str: return ""
def impl(arg: str) -> str: return ""
f.register(str)(impl)  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
"#,
);

// An overloaded impl registered with bare `@f.register` is still subtype-checked against the
// fallback (the dispatch type comes from the impl's first parameter).
functools_testcase!(
    test_singledispatch_register_overloaded_impl_subtype_checked,
    r#"
from functools import singledispatch
from typing import overload
@singledispatch
def f(arg: int) -> str: return ""
@overload
def g(arg: str) -> str: ...
@overload
def g(arg: str, y: int) -> str: ...
@f.register  # E: Dispatch type `str` is not a subtype of fallback first argument type `int`
def g(arg: str, y: int = 0) -> str: return ""
"#,
);

// The bare-functional signature-preserving path applies only to exactly `register(impl)`; a stray
// keyword falls through to the stub, which rejects it.
functools_testcase!(
    test_singledispatch_register_bare_functional_extra_kwarg,
    r#"
from functools import singledispatch
@singledispatch
def f(arg: object) -> str: return "base"
def impl(arg: int) -> str: return "int"
f.register(impl, foo=1)  # E: No matching overload found for function `functools.register`
"#,
);
