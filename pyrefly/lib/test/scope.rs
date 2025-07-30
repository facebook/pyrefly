/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_method_cannot_see_class_scope,
    r#"
class C:
    x: int

    def m(self) -> None:
        x  # E: Could not find name `x`
"#,
);

testcase!(
    test_more_class_scope,
    r#"
x: int = 0
class C:
    x: str = x # E: `int` is not assignable to `str`
    y: int = x # E: `str` is not assignable to `int`
    def m(self) -> str:
        # x refers to global x: int
        return x # E: Returned type `int` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_global_simple,
    r#"
x: str = ""
def f():
  global x
  x = "foo"
"#,
);

testcase!(
    test_global_ref_before_def,
    r#"
def f():
    global x
    x = "foo"
x: str = ""
"#,
);

testcase!(
    test_global_not_found,
    r#"
x: str = ""
global a  # E: Could not find name `a`
"#,
);

testcase!(
    test_global_assign_before_def,
    r#"
x: str = ""
global x  # E: `x` was assigned in the current scope before the global declaration
"#,
);

testcase!(
    test_global_aug_assign,
    r#"
x: str = ""
def f():
    x += "a"  # E: `x` is not mutable from the current scope
def g():
    global x
    x += "a"
def h0():
    global x
    def h1():
        x += "a"   # E: `x` is not mutable from the current scope
"#,
);

// Note: the root cause of behavior in this test is that if there is no assignment, we ignore the
// mutable capture entirely (leading to an error saying the mutation is invalid), whereas
// if there is an assignment we incorrectly mark the name as defined in the local scope.
testcase!(
    bug = "We fail to mark a del of a mutable capture as illegal",
    test_global_del,
    r#"
x: str = ""
def f():
    global x
    x = "foo"
    del x  # Not okay: it will work at runtime, but is not statically analyzable
# A minor variation on f(), relevant to specific implementation bugs in our scope analysis
# Here we do error, but for the wrong reason - we think `x` is an immutable capture.
def g():
    global x
    del x
f()
f()  # This will crash at runtime!
"#,
);

testcase!(
    bug = "It is not safe to treat global as a normal flow-sensitive definition",
    test_unannotated_global_with_reassignment,
    r#"
from typing import assert_type, Literal
x = "x"
def f():
    global x
    x = 42  # Should be a type error, does not respect module-level static analysis
    assert_type(x, Literal[42])
f()
# This is why allowing the reassignment is unsafe
assert_type(x, Literal['x'])
"#,
);

testcase!(
    test_global_assign_incompatible,
    r#"
x: str = ""
def f():
    global x
    x = 1  # E: `Literal[1]` is not assignable to variable `x` with type `str`
def g():
    x = 1  # OK, this is a new x
"#,
);

testcase!(
    test_global_after_local_define,
    r#"
x: str = ""
def f() -> None:
    y: int = 1
    global y  # E: `y` was assigned in the current scope before the global declaration
"#,
);

testcase!(
    bug = "We fail to add x to the global scope so that it's visible from `outer`",
    test_global_can_see_past_enclosing_scopes,
    r#"
from typing import assert_type
x: str = ""
def outer():
    x: int = 5
    def f():
        # This works fine in Python, `x` is the global `x`, not the one from `outer`.
        global x # E: Found `x`, but it was not the global scope
        assert_type(x, str)  # E: assert_type(Any, str)
"#,
);

testcase!(
    bug = "We currently never complain on aug assign, but when we fix it we need to be careful about type changes",
    test_global_aug_assign_incompatible_type,
    r#"
from typing import assert_type
class C:
    def __iadd__(self, other: C) -> C: ...
    def __sub__(self, other: C) -> C: ...
    def __mul__(self, other: C) -> int: ...
c0, c1, c2 = C(), C(), C()
def f():
    global c0
    global c1
    global c2
    # Should be permitted, the resulting operation is in-place
    c0 += C()
    # Should be permitted, the resulting operation returns a new C which is okay
    c1 -= C()
    # Should *not* be permitted, this changes the type of the global in a way
    # that is incompatible with static analysis of the global scope
    c2 *= C()
f()
# This shows what would go wrong if we allow the aug assign on `c2`
assert_type(c2, C)
"#,
);

testcase!(
    test_global_reference_in_nested_function,
    r#"
x: str = ""
def f() -> None:
    global x
    def g() -> str:
        return x
"#,
);

testcase!(
    test_nonlocal_simple,
    r#"
def f(x: int) -> None:
    def g():
        nonlocal x
        x = 1
"#,
);

testcase!(
    test_nonlocal_ref_before_def,
    r#"
def f(x: int) -> None:
    def g():
        nonlocal x
        x = 1
"#,
);

testcase!(
    test_nonlocal_not_found,
    r#"
def f() -> None:
    def g() -> None:
        nonlocal a
        a = "foo"
    a: str = ""
"#,
);

testcase!(
    test_nonlocal_assign_before_def,
    r#"
def f() -> None:
    a: str = ""
    nonlocal a  # E: `a` was assigned in the current scope before the nonlocal declaration
"#,
);

testcase!(
    test_nonlocal_aug_assign,
    r#"
def outer():
    x: str = ""
    def f():
        x += "a"  # E: `x` is not mutable from the current scope
    def g():
        nonlocal x
        x += "a"
    def h0():
        nonlocal x
        def h1():
            x += "a"   # E: `x` is not mutable from the current scope
"#,
);

// Note: the root cause of behavior in this test is that if there is no assignment, we ignore the
// mutable capture entirely (leading to an error saying the mutation is invalid), whereas
// if there is an assignment we incorrectly mark the name as defined in the local scope.
testcase!(
    bug = "We fail to mark a del of a mutable capture as illegal",
    test_nonlocal_del,
    r#"
def outer():
    x: str = ""
    def f():
        nonlocal x
        x = "foo"
        del x  # not okay: it will work at runtime, but is not statically analyzable
    # A minor variation on f(), relevant to specific implementation bugs in our scope analysis
    def g():
        nonlocal x
        del x
    f()
    f()  # This will crash at runtime!
"#,
);

testcase!(
    bug = "It is not safe to treat nonlocal as a normal flow-sensitive definition",
    test_unannotated_nonlocal_with_reassignment,
    r#"
from typing import assert_type, Literal
def outer():
    x = "x"
    def f():
        nonlocal x
        x = 42  # Should be a type error, does not respect parent scope typing
        assert_type(x, Literal[42])
    f()
    # This is why allowing reassignment with a different type is unsafe
    assert_type(x, Literal['x'])
"#,
);
testcase!(
    test_nonlocal_assign_incompatible,
    r#"
def f() -> None:
    a: str = ""
    def g() -> None:
        nonlocal a
        a = 1  # E: `Literal[1]` is not assignable to variable `a` with type `str`
    def h() -> None:
        a = 1  # OK, this is a new a
"#,
);

testcase!(
    test_nonlocal_multiple_annotations,
    r#"
def f() -> None:
    a: str = ""
    def g() -> None:
        nonlocal a
        a: int = 1  # E: `a` cannot be annotated with `int`, it is already defined with type `str`
"#,
);

testcase!(
    test_nonlocal_not_in_local_scope,
    r#"
x: str = ""
def f() -> None:
    nonlocal x  # E: Found `x`, but it was not in a valid enclosing scope
"#,
);

testcase!(
    test_nonlocal_reference_in_nested_function,
    r#"
def f() -> None:
    x: str = ""
    def g() -> None:
        nonlocal x
        def h() -> str:
            return x
"#,
);

testcase!(
    test_del_name,
    r#"
x: int
x + 1  # E: `x` is uninitialized
x = 1
x + 1  # OK
del x
x + 1  # E: `x` is uninitialized

y = 1
y + 1  # OK
del y
y + 1  # E: `y` is uninitialized

# check that we don't fall back to Any when the variable is annotated
z: int
z = str(z)  # E: `z` is uninitialized  # E: `str` is not assignable to variable `z` with type `int`
"#,
);

testcase!(
    test_uninitialized_merge_flow,
    r#"
def test(cond: bool):
    if cond:
        a: int
    else:
        a = 1
    a  # E: `a` may be uninitialized
    if cond:
        b: int
    else:
        b = 1
        del b
    b  # E: `b` is uninitialized
    if cond:
        c: int
    else:
        c: int = 1
    c  # E: `c` may be uninitialized
    if cond:
        d = 1
    else:
        d = 1
        del d
    d  # E: `d` may be uninitialized
    if cond:
        e = 1
    else:
        e: int
    e  # E: `e` may be uninitialized
    if cond:
        f = 1
        del f
    else:
        f: int
    f  # E: `f` is uninitialized
    if cond:
        g = 1
        del g
    else:
        g = 1
    g  # E: `g` may be uninitialized
    if cond:
        h = 1
        del h
    else:
        h = 1
        del h
    h  # E: `h` is uninitialized
"#,
);

testcase!(
    test_local_defined_by_mutation_no_shadowing,
    r#"
def f() -> None:
    x += 1  # E: Could not find name `x`
    del y  # E: Could not find name `y`
"#,
);

testcase!(
    bug = "We incorrectly treat mutations as potentially modifying non-mutable captures",
    test_local_defined_by_mutation_with_shadowing,
    r#"
x: int = 0
y: int = 0
def f() -> None:
    # This is *not* an invalid mutation of the global `x` (that is impossible syntactically).
    # Instead, it is *defining* a local (and should produce an uninitialaized local error in this case).
    # The runtime error is pretty clear here: UnboundLocalError: local variable 'x' referenced before assignment
    x += 1  # E: `x` is not mutable from the current scope
    # Same problem here, `del` defines a local, and the runtime gives the same error.
    del y  # E: `y` is not mutable from the current scope
"#,
);

testcase!(
    test_comprehension_shadows_variable,
    r#"
from typing import assert_type, reveal_type
x: list[int] = [1, 2, 3]
y = [x for x in x]
assert_type(y, list[int])
"#,
);

testcase!(
    test_walrus_behaviors,
    r#"
from typing import assert_type

def f(arg: int) -> None:
    x = (y := arg)
    assert_type(y, int)
    w = [z for x in [arg, arg] if (z := x) > 1]
    z  # E: Could not find name `z`
    assert_type(w, list[int])
    lambd = lambda x: (z := x) + 1
    z  # E: Could not find name `z`
"#,
);

testcase!(
    test_forward_reference_ok,
    r#"
def foo():
    x = y

y = 42
"#,
);

// https://github.com/facebook/pyrefly/issues/246
testcase!(
    test_declare_after_write_no_error,
    r#"
def foo():
    x = 42
    x: int
    y = x

def bar():
    x = 42
    x: int = -x
    y = x
"#,
);

// https://github.com/facebook/pyrefly/issues/154
testcase!(
    test_exception_not_in_scope,
    r#"
def try_except():
    try:
        1 / 0
    except Exception as e1:
        pass

    e1 # E: `e1` is uninitialized

def try_except_finally():
    try:
        1 / 0
    except Exception as e2:
        pass
    finally:
        e2 # E: `e2` is uninitialized

    e2 # E: `e2` is uninitialized

def try_except_twice():
    try:
        1 / 0
    except OSError as e3:
        pass
    except Exception:
        e3 # E: `e3` is uninitialized

    e3 # E: `e3` is uninitialized

def try_except_multiple_finally():
    try:
        1 / 0
    except OSError as e4:
        pass
    except Exception:
        pass
    except OSError as e5:
        e4 # E: `e4` is uninitialized
    finally:
        e4 # E: `e4` is uninitialized
        e5 # E: `e5` is uninitialized

    e4 # E: `e4` is uninitialized
    e5 # E: `e5` is uninitialized

def try_except_else():
    try:
        1 / 0
    except OSError as e6:
        pass
    except Exception:
        pass
    else:
        e6 # E: `e6` is uninitialized

    e6 # E: `e6` is uninitialized

def try_except_else_finally():
    try:
        1 / 0
    except OSError as e7:
        pass
    except Exception:
        pass
    else:
        e7 # E: `e7` is uninitialized
    finally:
        e7 # E: `e7` is uninitialized

    e7 # E: `e7` is uninitialized
"#,
);
