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
    test_method_can_access_dunder_class,
    r#"
from typing import assert_type
class Base:
    x: int
    def test(self):
        return __class__

    def test_nested(self):
        def inner():
            return __class__
        return inner()

assert_type(Base().test(), type[Base])
assert_type(Base().test_nested(), type[Base])
assert_type(Base().test().x, int)

class Child(Base):
    def test2(self):
        return __class__

assert_type(Child().test(), type[Base])
assert_type(Child().test2(), type[Child])
"#,
);

testcase!(
    test_nested_function_can_access_dunder_class,
    r#"
class Base:
    def show_class(self) -> type:
        def inner() -> type:
            return __class__
        return inner()
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
    global x  # E: `x` was assigned in the current scope before the global declaration
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
    test_initialize_on_usage,
    r#"
def test1(cond: bool) -> None:
    if cond:
        x: int
    else:
        x = 1
    print(x)  # E: `x` may be uninitialized
    print(x)

def test2(cond: bool) -> None:
    x: int
    if cond:
        print(x)  # E: `x` is uninitialized
    else:
        print(x)  # E: `x` is uninitialized
    print(x)

def test3(cond: bool) -> None:
    if cond:
        x: int
        print(x)  # E: `x` is uninitialized
    else:
        x = 1
    print(x)

def test4(cond: bool) -> None:
    x: int
    if cond:
        print(x)  # E: `x` is uninitialized
    else:
        x = 1
    print(x)
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
    assert_type(z, int)
    assert_type(w, list[int])
    lambd = lambda x: (z2 := x) + 1
    z2  # E: Could not find name `z2`
"#,
);

testcase!(
    test_walrus_in_comprehension_outer_scope,
    r#"
from typing import assert_type

def f() -> None:
    var = 33
    my_list = [var := str("some_str") for _ in [123]]
    assert_type(var, str)
    assert_type(my_list, list[str])

def g(xs: list[str]) -> None:
    last = ""
    results = [last := x for x in xs]
    assert_type(last, str)
    assert_type(results, list[str])

def h(matrix: list[list[int]]) -> None:
    result = [[y := x for x in row] for row in matrix]
    assert_type(y, int)
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

testcase!(
    test_exception_not_in_scope,
    r#"
def try_except():
    try:
        1 / 0 # E: Cannot divide by zero
    except Exception as e1:
        pass

    e1 # E: `e1` is uninitialized

def try_except_finally():
    try:
        1 / 0 # E: Cannot divide by zero
    except Exception as e2:
        pass
    finally:
        e2 # E: `e2` is uninitialized

    e2

def try_except_twice():
    try:
        1 / 0 # E: Cannot divide by zero
    except OSError as e3:
        pass
    except Exception:
        e3 # E: `e3` is uninitialized

    e3 # E: `e3` may be uninitialized

def try_except_multiple_finally():
    try:
        1 / 0 # E: Cannot divide by zero
    except OSError as e4:
        pass
    except Exception:
        pass
    except OSError as e5:
        e4 # E: `e4` is uninitialized
    finally:
        e4 # E: `e4` is uninitialized
        e5 # E: `e5` is uninitialized

    e4
    e5

def try_except_else():
    try:
        1 / 0 # E: Cannot divide by zero
    except OSError as e6:
        pass
    except Exception:
        pass
    else:
        e6 # E: `e6` is uninitialized

    e6 # E: `e6` may be uninitialized

def try_except_else_finally():
    try:
        1 / 0 # E: Cannot divide by zero
    except OSError as e7:
        pass
    except Exception:
        pass
    else:
        e7 # E: `e7` is uninitialized
    finally:
        e7 # E: `e7` may be uninitialized

    e7
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
    test_dunder_all_duplicate_augassign,
    r#"
__all__ += ["A"]  # E: `__all__` is uninitialized  # E: Name `A` is listed in `__all__` but is not defined in the module
__all__ += ["A"]  # E: Name `A` is listed in `__all__` but is not defined in the module
"#,
);

testcase!(
    test_aug_assign_lookup_inconsistencies,
    r#"
from typing import assert_type, Any
def f():
    assert_type(x, Any)  # E: `x` is uninitialized
    x += 5
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
    del y
    del x
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
        assert_type(x, str)
    lambda_f = lambda: assert_type(x, str)
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

testcase!(
    test_captured_var_in_nested_function_with_flow_merge,
    r#"
def test_annotated():
    x: dict[int, int]
    x = {}
    def nested(a: int, bs: list[int]):
        if not x[0]:
            return None
        for b in bs:
            break
        return x
def test_unannotated():
    x = {}
    def nested(a: int, bs: list[int]):
        if not x[0]:
            return None
        for b in bs:
            break
        return x
def test_walrus():
    x = {}
    def nested(a: int, bs: list[int]):
        if not (y := x)[0]:
            return None
        for b in bs:
            break
        return (z := x)
"#,
);

testcase!(
    test_captured_var_pins_outer,
    r#"
from typing import assert_type, Any

def test():
    x = []
    def nested():
        x.append(1)
    x.append("")  # E: Argument `Literal['']` is not assignable to parameter `object` with type `int` in function `list.append`

def test2():
    x = []
    def nested():
        x.append(1)
    assert_type(x[0], int)
"#,
);

testcase!(
    test_captured_var_narrow,
    r#"
from typing import assert_type
def test():
    x: int | None = 1
    def nested() -> int:
        if x is not None:
            assert_type(x, int)
            return x
        else:
            assert_type(x, None)
            return 0
"#,
);

testcase!(
    test_no_capture_class_var,
    r#"
# We should not capture variables from the class body
from typing import assert_type
x: int
class C:
    x: str
    def test(self):
        assert_type(x, int)
"#,
);

testcase!(
    test_captured_var_no_uninitialized_error,
    r#"
# If a variable is captured from outside, we should not give an error that it's uninitialized, even if it is not
def test():
    x: int
    def nested() -> int:
        return x
"#,
);

testcase!(
    test_capture_var_preserves_import_flow_style,
    r#"
from typing import assert_type
x: object = 1
def test(y: object):
    while True:
        z: object = 1
        if hasattr(x, "foo"):
            x.foo
        if hasattr(y, "foo"):
            y.foo
        if hasattr(z, "foo"):
            z.foo
"#,
);

testcase!(
    // #1804: is not None guard, not reassigned
    test_narrow_capture_is_not_none,
    r#"
from typing_extensions import assert_type
def f(x: int | None) -> None:
    if x is not None:
        assert_type(x, int)
        def g() -> int:
            assert_type(x, int)
            return x + 1
        g()
"#,
);

testcase!(
    // #1804: x is reassigned so the narrow does NOT propagate
    test_narrow_capture_reassigned_after,
    r#"
def f_reassigned(x: int | None) -> None:
    if x is not None:
        def g() -> int:
            return x + 1  # E: `+` is not supported between `None` and `Literal[1]`
        x = None
        g()
"#,
);

testcase!(
    // #2394: Callable | None narrowing in nested scope
    test_narrow_capture_callable,
    r#"
from typing import Callable
from typing_extensions import assert_type
def process(key: Callable[[str], str] | None) -> None:
    if key is not None:
        assert_type(key, Callable[[str], str])
        def inner() -> str:
            assert_type(key, Callable[[str], str])
            return key("value")
        inner()
"#,
);

testcase!(
    // #2513: Early-return guard
    test_narrow_capture_early_return,
    r#"
from typing_extensions import assert_type
def process_name(name: str) -> None: ...
def handle_request(name: str | None, use_callback: bool) -> None:
    if name is None:
        return
    assert_type(name, str)
    def callback() -> None:
        assert_type(name, str)
        process_name(name)
    if use_callback:
        callback()
"#,
);

testcase!(
    // #919: isinstance narrowing in nested function
    test_narrow_capture_isinstance,
    r#"
from typing_extensions import assert_type
class A:
    def foo(self) -> None: pass
def get_a() -> object:
    return A()
def bar() -> None:
    a: object = get_a()
    assert isinstance(a, A)
    assert_type(a, A)
    a.foo()
    def foobar() -> None:
        assert_type(a, A)
        a.foo()
"#,
);

testcase!(
    // #768: assert is not None after loop (needs last_range check)
    test_narrow_capture_assert_after_loop,
    r#"
from typing_extensions import assert_type
def f(rows: list[int]) -> int:
    table_start: int | None = None
    for r in rows:
        if table_start is None:
            table_start = r
    assert table_start is not None
    assert_type(table_start, int)
    def inner() -> int:
        assert_type(table_start, int)
        return table_start - 1
    return inner()
"#,
);

testcase!(
    // #2739: reassignment before nested function definition should preserve the reassigned type
    test_narrow_capture_reassigned_before_nested_def,
    r#"
from typing import Callable
from typing_extensions import assert_type

class Connection:
    def __init__(self, host: str) -> None:
        self.host = host

    def clone(self) -> "Connection":
        return Connection(self.host)

def get_connection() -> Connection | None:
    return Connection("localhost")

def make_query_func() -> Callable[[], str]:
    conn = get_connection()
    assert conn is not None
    conn = conn.clone()
    assert_type(conn, Connection)

    def query() -> str:
        assert_type(conn, Connection)
        return conn.host

    return query
"#,
);

testcase!(
    // #2408: isinstance narrowing with derived variable
    test_narrow_capture_isinstance_derived,
    r#"
import copy
from typing_extensions import assert_type
def run(code: str | bytes) -> None:
    if isinstance(code, str):
        string = copy.copy(code)
        assert_type(code, str)
        assert_type(string, str)
        def run_code1() -> None:
            assert_type(code, str)
            exec(code)
        def run_code2() -> None:
            assert_type(string, str)
            exec(string)
"#,
);

testcase!(
    // #765: Lambda captures already see outer narrows
    test_narrow_capture_lambda,
    r#"
from typing import Callable
from typing_extensions import assert_type
def foo(obj: str | None) -> Callable[[], str]:
    if obj is None:
        return lambda: "default"
    assert_type(obj, str)
    return lambda: obj + "bar"
"#,
);

testcase!(
    // #1800: Final variable narrowing at module level
    test_narrow_capture_final_module_level,
    r#"
from typing import Final
from typing_extensions import assert_type
param: Final[str | None] = ""
if param is None:
    raise ValueError()
assert_type(param, str)
def foo() -> None:
    assert_type(param, str)
"#,
);

testcase!(
    // #40: Walrus operator narrowing in nested function
    test_narrow_capture_walrus,
    r#"
from typing import Callable
from typing_extensions import assert_type
class Foo:
    _window_function: Callable[[str], int] | None
    def foo(self) -> None:
        if (window_function := self._window_function):
            assert_type(window_function, Callable[[str], int])
            def bar() -> None:
                assert_type(window_function, Callable[[str], int])
                window_function("foo")
            bar()
"#,
);

// Adapted from https://github.com/pypa/pip/blob/main/src/pip/_vendor/pygments/console.py
testcase!(
    test_narrow_capture_regression_1,
    r#"
esc = "\x1b["
codes = {}
codes[""] = ""
codes["bold"] = esc + "01m"
codes["white"] = codes["bold"]

def ansiformat(attr, text):
    result = []
    if attr[:1] == attr[-1:] == '*':
        result.append(codes['bold'])
        attr = attr[1:-1]
    result.append(codes[attr])
    result.append(text) # false positive: str not assignable to LiteralString
    result.append(codes['reset'])
    return ''.join(result)
"#,
);

// Adapted from https://github.com/home-assistant/core/blob/dev/homeassistant/components/plex/server.py
testcase!(
    test_narrow_capture_control_flow_narrow,
    r#"
from typing import Literal, assert_type
def test_1(cond: bool, x: int):
    active_session = x if cond else None
    if not active_session:
        return
    def update_with_new_media():
        assert_type(active_session, int)
def test_2(cond: bool):
    active_session: int | None = 1 if cond else None
    if not active_session:
        return
    def update_with_new_media():
        assert_type(active_session, int)
"#,
);

// https://github.com/facebook/pyrefly/issues/2930
testcase!(
    bug = "del in dead code should still make the variable local, hiding the outer binding",
    test_del_in_dead_code_makes_local,
    r#"
x = 1

def f():
    # Even though the del is in unreachable code, Python's compiler
    # still marks x as local in f's scope.
    print(x)
    if False:
        del x
"#,
);

// https://github.com/facebook/pyrefly/issues/2929
testcase!(
    bug = "Should detect possibly-missing attribute on conditionally-defined class members",
    test_conditionally_defined_class_member,
    r#"
def coin() -> bool:
    return True

class Config:
    name: str = "default"
    if coin():
        debug = True
"#,
);
