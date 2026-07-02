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
    bug = "pyrefly does not check that the dispatch type passed to .register() is a subtype of the fallback function's first argument",
    test_singledispatch_register_explicit_type_not_subtype,
    r#"
from functools import singledispatch

@singledispatch
def f(arg: int) -> None:
    pass

# WANT: Dispatch type "str" must be subtype of fallback function first argument "int"
@f.register(str)
def g(arg) -> None:
    pass
"#,
);

functools_testcase!(
    bug = "pyrefly does not check that the @register dispatch type is a subtype of the fallback's first argument",
    test_singledispatch_register_custom_class_not_subtype,
    r#"
from functools import singledispatch
class A: pass

@singledispatch
def f(arg: int) -> None:
    pass
@f.register(A)
# WANT: Dispatch type "A" must be subtype of fallback function first argument "int"
def g(arg) -> None:
    pass
"#,
);

functools_testcase!(
    bug = "pyrefly does not check that a @register'd dispatch type is a subtype of the fallback function's first argument",
    test_singledispatch_register_impl_type_not_subtype,
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

@f.register
# WANT: Dispatch type "C" must be subtype of fallback function first argument "A"
def h(arg: C) -> None:
    pass
"#,
);

functools_testcase!(
    bug = "pyrefly does not check that the register dispatch type is a subtype of the fallback function's first argument type",
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

# WANT: Dispatch type "ConstExpr" must be subtype of fallback function first argument "Base"
@f.register(ConstExpr)
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

functools_testcase!(
    bug = "pyrefly does not check argument types when calling singledispatch-registered impls directly",
    test_singledispatch_registered_impl_direct_call_arg_check,
    r#"
from functools import singledispatch

@singledispatch
def f(arg, arg2: str) -> bool:
    return False

@f.register
def g(arg: int, arg2: str) -> bool:  # E: Function declared to return `bool` but is missing an explicit `return`
    pass

@f.register(str)
def h(arg, arg2: str) -> bool:  # E: Function declared to return `bool` but is missing an explicit `return`
    pass

# WANT: Argument 1 to "g" has incompatible type "str"; expected "int"
g('a', 'a')
# WANT: Argument 2 to "g" has incompatible type "int"; expected "str"
g(1, 1)

# don't show errors for incorrect first argument here, because there's no type annotation for the
# first argument
h(1, 'a')
# WANT: Argument 2 to "h" has incompatible type "int"; expected "str"
h('a', 1)
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
    bug = "registered impl dispatch type not a subtype of fallback first arg should error",
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
@f.register
# WANT: Dispatch type `C` is not a subtype of fallback first argument type `A`
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
