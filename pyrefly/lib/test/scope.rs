/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every SPELL deliberately testing bad spelling

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
    test_unknown_name_suggests_similar,
    r#"
long_variable_name = 1
long_variable_name2 = long_variuble_name  # E: Did you mean `long_variable_name`?
"#,
);

testcase!(
    test_unknown_name_suggests_from_enclosing_scope,
    r#"
outer_value = 10
def f() -> int:
    return outer_vlaue  # E: Did you mean `outer_value`?
"#,
);

testcase!(
    test_unknown_name_no_suggest_from_future_defs,
    r#"
future_value = missing  # E: `missing` is uninitialized  # !E: Did you mean
missing = 1
"#,
);

testcase!(
    test_unknown_name_no_suggest_from_class_scope_in_method,
    r#"
class C:
    x = 1
    def m(self) -> int:
        return x  # E: Could not find name `x`  # !E: Did you mean
"#,
);

testcase!(
    test_unknown_name_suggests_in_class_body,
    r#"
class Foo:
    abc = 42
    y = ab + 42  # E: Did you mean `abc`?
"#,
);

testcase!(
    test_unknown_name_no_suggest_single_letter_names,
    r#"
a = 1
b = 2
aa  # E: Could not find name `aa`  # !E: Did you mean
"#,
);

testcase!(
    test_unknown_name_prefers_inner_scope,
    r#"
value = 0
def f() -> int:
    local_value = 1
    return local_valu  # E: Did you mean `local_value`?
"#,
);

testcase!(
    test_unknown_name_ties_prefer_shallower_scope,
    r#"
global_value = 1
def outer() -> int:
    global_value2 = 2
    def inner() -> int:
        globl_value = 3
        return globl_valu  # E: Did you mean `globl_value`?
    return inner()
"#,
);

testcase!(
    test_unknown_name_no_suggestion_when_far,
    r#"
alpha = 1
beta = 2
missing_completely = gamma  # E: Could not find name `gamma`  # !E: Did you mean
"#,
);

// The python compiler enforces static scoping rules in most cases - for example, if a function
// defines a name and we try to read it before we write it, Python will normally not fall back
// to searching in enclosing scopes.
//
// But class body scopes are dynamic - Python just checks the currently-defined
// locals and then keeps looking. This allows Python developers to do things
// like shadow a global in a class body with a simple assignment where the RHS
// uses the name being defined in the LHS.
testcase!(
    test_class_scope_is_dynamic,
    r#"
x: int = 0
z: int = 0
class C:
    z: str = str(z)
    x: str = x # E: `int` is not assignable to `str`
    y: int = x # E: `str` is not assignable to `int`
    # Inside of a method, x refers to the global x: int
    def m(self) -> str:
        return x # E: Returned type `int` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_await_outside_async_function_def,
    r#"
async def make_int() -> int:
    return 1
await make_int()  # E: `await` can only be used inside an async function
def test():
    await make_int()  # E: `await` can only be used inside an async function
async def test_async():
    await make_int()  # ok
"#,
);

testcase!(
    test_async_for_outside_async_function_def,
    r#"
import asyncio
class AsyncIterator:
    def __init__(self):
        pass
    def __aiter__(self):
        return self
    async def __anext__(self):
        raise StopAsyncIteration
async for value in AsyncIterator(): pass  # E: `async for` can only be used inside an async function
def test():
    async for value in AsyncIterator(): pass  # E: `async for` can only be used inside an async function
async def test_async():
    async for value in AsyncIterator(): pass  # ok
"#,
);

testcase!(
    test_async_with_outside_async_function_def,
    r#"
import asyncio
class AsyncContextManager:
    async def __aenter__(self):
        print("Entered context")
        return self
    async def __aexit__(self, exc_type, exc, tb):
        print("Exited context")
async with AsyncContextManager(): pass  # E: `async with` can only be used inside an async function
def test():
    async with AsyncContextManager(): pass  # E: `async with` can only be used inside an async function
async def test_async():
    async with AsyncContextManager(): pass  # ok
"#,
);

testcase!(
    test_await_and_async_comprehensions,
    r#"
from typing import Any

# A bare (parenthesized) generator containing an await immediately produces an
# AsyncGenerator[_, _] result. It is legal to use in a synchronous function, although
# it cannot be iterated except in an async function.
#
# Other kinds of comprehensions (list, set, etc) cannot use `await` unless in an async
# function.

def test(xs: Any):
    (await x for x in xs)  # Ok
    [await x for x in xs]  # E:
    {await x for x in xs}  # E:
    {0: await x for x in xs}  # E:
    {await x: 0 for x in xs}  # E:

    (x async for x in xs)  # OK
    [x async for x in xs]  # E:
    {x async for x in xs}  # E:
    {x: 0 async for x in xs}  # E:

    (x for x in await xs)  # E:
    [x for x in await xs]  # E:
    {x for x in await xs}  # E:
    {x: 0 for x in await xs}  # E:

    (await x async for x in (await y async for y in xs))  # Ok
    [await x async for x in (await y async for y in xs)]  # E: `async` # E: `await`
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
    test_nonlocal_simple,
    r#"
def f(x: int) -> None:
    def g():
        nonlocal x
        x = 1
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
    test_nonlocal_ref_before_def,
    r#"
def f(x: int) -> None:
    def g():
        nonlocal x
        x = 1
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
    test_global_can_see_past_enclosing_scopes,
    r#"
from typing import assert_type
x: str = ""
def outer():
    x: int = 5
    def f():
        global x
        assert_type(x, str)
"#,
);

testcase!(
    test_nonlocal_finds_global,
    r#"
from typing import Any, assert_type
x: str = ""
def f() -> None:
    nonlocal x  # E: Found `x`, but it is coming from the global scope
def outer():
    x: int = 5
    def middle():
        global x
        assert_type(x, str)
        def inner():
            nonlocal x  # E: Found `x`, but it is coming from the global scope
            assert_type(x, Any)
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
    test_mutable_capture_assign_before_def,
    r#"
def f() -> None:
    a: str = ""
    nonlocal a  # E: `a` was assigned in the current scope before the nonlocal declaration
"#,
);

// Note: the root cause of behavior in this test is that if there is no assignment, we ignore the
// mutable capture entirely (leading to an error saying the mutation is invalid), whereas
// if there is an assignment we incorrectly mark the name as defined in the local scope.
testcase!(
    bug = "We fail to mark a del of a mutable capture as illegal",
    test_mutable_capture_del,
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
    test_unannotated_mutable_capture_with_reassignment,
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
    test_mutable_capture_assign_incompatible,
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
    test_mutable_capture_multiple_annotations,
    r#"
def f() -> None:
    a: str = ""
    def g() -> None:
        nonlocal a
        a: int = 1  # E: `a` cannot be annotated with `int`, it is already defined with type `str`
"#,
);

testcase!(
    test_mutable_capture_reference_in_nested_function,
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
    test_mutable_capture_class_body,
    r#"
from typing import assert_type
x = 5
class C:
    global x
    x = 7
C().x  # E: `C` has no attribute `x`
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1210
testcase!(
    test_mutable_capture_with_annotation_class_body,
    r#"
x = 5
class C:
    global x
    x: int
"#,
);

testcase!(
    bug = "We allow mutable captures to mutate in ways that invalidate through-barrier types on globals and nonlocals",
    test_mutable_capture_incompatible_assign,
    r#"
from typing import reveal_type
x = 5
def f():
    global x
    reveal_type(x)  # E: revealed type: Literal['str', 5]
    x = b'bytes'  # This ought to be an error (unless the reveal_type above were to account for the mutation)
x = "str"
"#,
);

testcase!(
    bug = "We detect the error (we didn't at one point) but the error message is not very good",
    test_mutable_capture_read_before_declared,
    r#"
x = 42
def f():
    # We should really be producing an error more like the compiler's, which says you can't use `x` before the declaration
    print(x)  # E: `x` is uninitialized
    global x
"#,
);

// Regression test against a panic in the presence of parser error recovery:
// https://github.com/facebook/pyrefly/issues/1203
testcase!(
    test_mutable_capture_syntax_error,
    r#"
def f():
    be be:  # E: `be` is uninitialized  # E: Parse error  # E: Parse error
    global be   # E: `be` was assigned in the current scope before the global declaration
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
    test_uninitialized_when_shadowing,
    r#"
from typing import assert_type
x: int = 5
def f():
    assert_type(x, str)  # E: `x` is uninitialized
    x: str = "foo"
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
    x += 1  # E: `x` is uninitialized
    del y  # E: `y` is uninitialized
"#,
);

testcase!(
    test_local_defined_by_mutation_with_shadowing,
    r#"
x: int = 0
y: int = 0
def f() -> None:
    x += 1  # E: `x` is uninitialized
    del y  # E: `y` is uninitialized
"#,
);

testcase!(
    test_comprehension_shadows_variable,
    r#"
from typing import assert_type
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

// https://github.com/facebook/pyrefly/issues/959
testcase!(
    test_lambda_captures_narrowed_variable,
    r#"
class A:
    shape: int = 0

def check[T](new: T, old: T) -> None:
    if isinstance(new, A):
        assert isinstance(old, A)
        lambda: old.shape
"#,
);

testcase!(
    test_dunder_all_mutated_without_def,
    r#"
__all__ += []  # E: `__all__` is uninitialized
"#,
);

testcase!(
    test_aug_assign_lookup_inconsistencies,
    r#"
from typing import assert_type, Any
def f():
    assert_type(x, Any)  # E: `x` is uninitialized
    x += 5  # E: `x` is uninitialized
    assert_type(x, Any)
"#,
);

testcase!(
    test_del_defines_a_local,
    r#"
from typing import Any, assert_type
x = 5
def f():
    assert_type(y, Any)  # E: `y` is uninitialized
    assert_type(x, Any)  # E: `x` is uninitialized
    del y  # E: `y` is uninitialized
    del x  # E: `x` is uninitialized
f()
"#,
);

testcase!(
    test_parameter_is_only_deleted_by_body,
    r#"
def fun(arg):
    def inner():
        arg  # The capture here makes sure we don't panic on a bad key lookup
    del arg
    return inner
    "#,
);

testcase!(
    test_parameter_is_declared_as_mutable_capture,
    r#"
x = 42
def fun(x):
    def inner():
        x  # The capture here makes sure we don't panic on a bad key lookup
    global x  # E: `x` was assigned in the current scope before the global declaration
    return inner
    "#,
);

testcase!(
    test_type_statement_scope,
    r#"
from typing import assert_type
class A: pass
type X[A] = list[A]
assert_type(A, type[A])
    "#,
);

// This does come up in practice - see https://github.com/facebook/pyrefly/issues/1146
testcase!(
    test_class_scope_annotation_shadows_global,
    r#"
class A: pass
class B:
    # This sets `A` in `__annotations__`, but because class scopes are dynamic,
    # `A` still refers to the global.
    A: A
    x: A = A()
class C:
    # The use of `del` is more of an edge case, but our implementation has to
    # define the behavior and we should test it. It behaves the same.
    A = A()
    del A
    y: A = A()
"#,
);

testcase!(
    test_class_scope_annotation_shadows_function,
    r#"
class D:
    def int(self) -> None:
        ...
    y: int = 0  # E: Expected a type form
"#,
);

// Nested scopes - except for parameter scopes - cannot see a containing class
// body. This applies not only to methods but also other scopes like lambda, inner
// class bodies, and comprehensions. See https://github.com/facebook/pyrefly/issues/264
testcase!(
    test_class_scope_lookups_when_skip,
    r#"
from typing import assert_type, Literal
x = 'string'
class A:
    x = 42
    def f():
        assert_type(x, Literal['string'])
    lambda_f = lambda: assert_type(x, Literal['string'])
    class B:
        assert_type(x, Literal['string'])
    [assert_type(x, Literal['string']) for _ in range(1)]
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1073 and
// https://github.com/facebook/pyrefly/issues/1074
testcase!(
    test_class_scope_lookups_when_permitted,
    r#"
# There are some cases where we want to be sure class body lookup is permitted:
# - Comprehension iterators are evaluated in the parent scope, not the comprehension
#   scope, so it is legal to use class body vars.
# - Method parameters are defined in a scope that *can* see the class body,
#   so parameter defaults may use class body vars.
class C:
    X = "X"
    x_chars = [char for char in X]
    def __init__(self, x: str = X):
        self.x = x
    "#,
);

testcase!(
    bug = "We don't yet handle all class body members correctly",
    test_class_scope_edge_cases,
    r#"
from typing import assert_type, Any

# The leftmost iterator of a comprehension evaluates in the parent scope, but
# other iterators evaluate in the comprehension scope. This behavior is only really
# evident in class bodies - this test checks that we get it exactly right.
class A:
    xs = [1, 2, 3]
    ys = [(a, b) for a in xs for b in ["a", "b"]]
    zs = [(a, b) for a in ["a", "b"] for b in xs]  # E: Could not find name `xs`
assert_type(A.ys, list[tuple[int, str]])
assert_type(A.zs, list[tuple[str, Any]])

# The visibility rules should cover all elements of the class scope (at one point
# implementation detatils made it depend on the flow style, so that we only
# caught cases involving assignment but not other kinds of definitions).
class B:
    @staticmethod
    def f(): pass
    cb = lambda _: f()  # E: Could not find name `f`

# The visibility rules should understand scope layering - an annotation scope
# can only see the containing class body, not any class body further up the scope stack
class C:
    x = 5
    class Inner:
        def g(self, z = x):  # E: Could not find name `x`
            pass
    "#,
);

testcase!(
    test_global_in_inner_function,
    r#"
from typing import assert_type

x: int = 1

def outer():
    x: str = ""

    def inner():
        global x
        assert_type(x, int)
    "#,
);

testcase!(
    test_global_with_same_name_as_local,
    r#"
from typing import assert_type, Literal

x = 42

def f():
    global x
    assert_type(x, Literal[42])
    x = "foo"
    "#,
);
