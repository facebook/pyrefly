/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Write;

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;
use crate::testcase;

testcase!(
    test_if_simple,
    r#"
from typing import assert_type, Literal
def b() -> bool:
    return True
if b():
    x = 100
else:
    x = "test"
y = x
assert_type(y, Literal['test', 100])
"#,
);

testcase!(
    test_if_else,
    r#"
from typing import assert_type, Literal
def b() -> bool:
    return True
if b():
    x = 100
elif b():
    x = "test"
else:
    x = True
y = x
assert_type(y, Literal['test', 100, True])
"#,
);

testcase!(
    test_if_only,
    r#"
from typing import assert_type, Literal
def b() -> bool:
    return True
x = 7
if b():
    x = 100
y = x
assert_type(y, Literal[7, 100])
"#,
);

testcase!(
    test_listcomp_simple,
    r#"
from typing import assert_type
y = [x for x in [1, 2, 3]]
assert_type(y, list[int])
    "#,
);

testcase!(
    test_listcomp_no_leak,
    r#"
def f():
    y = [x for x in [1, 2, 3]]
    return x  # E: Could not find name `x`
    "#,
);

testcase!(
    test_listcomp_no_overwrite,
    r#"
from typing import assert_type
x = None
y = [x for x in [1, 2, 3]]
assert_type(x, None)
    "#,
);

testcase!(
    test_listcomp_read_from_outer_scope,
    r#"
from typing import assert_type
x = None
y = [x for _ in [1, 2, 3]]
assert_type(y, list[None])
    "#,
);

testcase!(
    test_listcomp_iter_error,
    r#"
class C:
    pass
[None for x in C.error]  # E: Class `C` has no class attribute `error`
    "#,
);

testcase!(
    test_listcomp_if_error,
    r#"
class C:
    pass
def f(x):
    [None for y in x if "5" + 5]  # E: `+` is not supported between `Literal['5']` and `Literal[5]`
    "#,
);

testcase!(
    test_listcomp_target_error,
    r#"
def f(x: list[tuple[int]]):
    [None for (y, z) in x]  # E: Cannot unpack
    "#,
);

testcase!(
    test_listcomp_splat,
    r#"
from typing import assert_type
def f(x: list[tuple[int, str, bool]]):
    z = [y for (_, *y) in x]
    assert_type(z, list[list[bool | str]])
    "#,
);

testcase!(
    test_setcomp,
    r#"
from typing import assert_type
y = {x for x in [1, 2, 3]}
assert_type(y, set[int])
    "#,
);

testcase!(
    test_dictcomp,
    r#"
from typing import assert_type
def f(x: list[tuple[str, int]]):
    d = {y: z for (y, z) in x}
    assert_type(d, dict[str, int])
    "#,
);

testcase!(
    test_generator,
    r#"
from typing import assert_type, Generator
y = (x for x in [1, 2, 3])
assert_type(y, Generator[int, None, None])
    "#,
);

testcase!(
    test_bad_loop_command,
    r#"
break  # E: `break` outside loop
continue  # E: `continue` outside loop
    "#,
);

testcase!(
    test_break,
    r#"
from typing import assert_type, Literal
def f(cond):
    x = None
    for i in [1, 2, 3]:
        x = i
        if cond():
            break
        x = "hello world"
    assert_type(x, Literal["hello world"] | int | None)
    "#,
);

testcase!(
    test_continue,
    r#"
from typing import assert_type, Literal
def f(cond1, cond2):
    x = None
    while cond1():
        x = 1
        if cond2():
            x = 2
            continue
        assert_type(x, Literal[1])
        x = "hello world"
    assert_type(x, Literal["hello world", 2] | None)
    "#,
);

testcase!(
    test_early_return,
    r#"
from typing import assert_type, Literal
def f(x):
    if x:
        y = 1
        return
    else:
        y = "2"
    assert_type(y, Literal["2"])
    "#,
);

// Regression test to ensure we don't forget to create loop recursion
// bindings when the loop has early termination.
testcase!(
    test_return_in_for,
    r#"
def f(x: str):
    for c in x:
        x = c
        return
    "#,
);

testcase!(
    test_flow_scope_type,
    r#"
from typing import assert_type

# C itself is in scope, which means it ends up bound to a Phi
# which can cause confusion as both a type and a value
class C: pass

c = C()

while True:
    if True:
        c = C()

assert_type(c, C)
    "#,
);

testcase!(
    test_flow_crash,
    r#"
def test():
    while False:
        if False:
            x: int
        else:
            x: int
            if False:
                continue
"#,
);

testcase!(
    test_flow_crash2,
    r#"
def magic_breakage(argument):
    for it in []:
        continue
        break
    else:
        raise
"#,
);

testcase!(
    test_try,
    r#"
from typing import assert_type, Literal

try:
    x = 1
except:
    x = 2

assert_type(x, Literal[1, 2])
"#,
);

testcase!(
    test_exception_handler,
    r#"
from typing import assert_type

class Exception1(Exception): pass
class Exception2(Exception): pass

x1: tuple[type[Exception], ...] = (Exception1, Exception2)
x2 = (Exception1, Exception2)

try:
    pass
except int as e1:  # E: Invalid exception class: `int` does not inherit from `BaseException`
    assert_type(e1, int)
except int:  # E: Invalid exception class
    pass
except Exception as e2:
    assert_type(e2, Exception)
except ExceptionGroup as e3:
    assert_type(e3, ExceptionGroup[Exception])
except (Exception1, Exception2) as e4:
    assert_type(e4, Exception1 | Exception2)
except Exception1 as e5:
    assert_type(e5, Exception1)
except x1 as e6:
    assert_type(e6, Exception)
except x2 as e7:
    assert_type(e7, Exception1 | Exception2)
"#,
);

testcase!(
    test_exception_handler_dynamic_tuple,
    r#"
from typing import assert_type

class Exception1(Exception): pass
class Exception2(Exception): pass

# Dynamic tuple from tuple() constructor call
error_list = [Exception1, Exception2]
dynamic_errors = tuple(error_list)
try:
    pass
except dynamic_errors as e1:
    assert_type(e1, Exception1 | Exception2)

# Union-typed parameter: single exception class or tuple of exception classes
def handle(
    errors: type[Exception] | tuple[type[Exception], ...],
    value: str,
) -> int:
    try:
        return int(value)
    except errors as e2:
        assert_type(e2, Exception)
        return 0
"#,
);

testcase!(
    test_exception_handler_star_unpacking,
    r#"
import sys
from typing import assert_type

EXTRA_ERRORS: tuple[type[Exception], ...] = (RuntimeError,) if sys.version_info < (3, 13) else ()

try:
    pass
except (ValueError, *EXTRA_ERRORS) as e:
    assert_type(e, ValueError | Exception)
"#,
);

testcase!(
    test_exception_group_handler,
    r#"
from typing import reveal_type

class Exception1(Exception): pass
class Exception2(Exception): pass

try:
    pass
except* int as e1:  # E: Invalid exception class
    reveal_type(e1)  # E: revealed type: ExceptionGroup[int]
except* Exception as e2:
    reveal_type(e2)  # E: revealed type: ExceptionGroup
except* ExceptionGroup as e3:  # E: Exception handler annotation in `except*` clause may not extend `BaseExceptionGroup`
    reveal_type(e3)  # E: ExceptionGroup[ExceptionGroup]
except* (Exception1, Exception2) as e4:
    reveal_type(e4)  # E: ExceptionGroup[Exception1 | Exception2]
except* Exception1 as e5:
    reveal_type(e5)  # E: ExceptionGroup[Exception1]
"#,
);

testcase!(
    test_try_else,
    r#"
from typing import assert_type, Literal

try:
    x = 1
except:
    x = 2
else:
    x = 3

assert_type(x, Literal[2, 3])
"#,
);

testcase!(
    test_try_finally,
    r#"
from typing import assert_type, Literal

try:
    x = 1
except:
    x = 2
finally:
    x = 3

assert_type(x, Literal[3])
"#,
);

testcase!(
    test_match,
    r#"
from typing import assert_type

def point() -> int:
    return 3

match point():
    case 1:
        x = 8
    case q:
        x = q
assert_type(x, int)
"#,
);

testcase!(
    test_match_narrow_simple,
    r#"
from typing import assert_type, Literal

def test(x: int):
    match x:
        case 1:
            assert_type(x, Literal[1])
        case 2 as q:
            assert_type(x, Literal[2])
            assert_type(q, Literal[2])
        case q:
            assert_type(x, int)
            assert_type(q, int)

x: object = object()
match x:
    case int():
        assert_type(x, int)

y: int | str = 1
match y:
    case str():
        assert_type(y, str)
"#,
);

testcase!(
    bug = "does not detect unreachable branches based on nested patterns",
    test_match_narrow_len,
    r#"
from typing import assert_type, Never

def foo(x: tuple[int, int] | tuple[str]):
    match x:
        case [x0]:
            assert_type(x, tuple[str])
            assert_type(x0, str)
    match x:
        case [x0, x1]:
            assert_type(x, tuple[int, int])
            assert_type(x0, int)
            assert_type(x1, int)
    match x:
        # these two cases should be impossible to match
        case [str(), str()]:
            assert_type(x, tuple[int, int])
        case [int()]:
            assert_type(x, tuple[str])
"#,
);

testcase!(
    test_match_mapping,
    r#"
from typing import assert_type

x: dict[str, int] = { "a": 1, "b": 2, "c": 3 }
match x:
    case { "a": 1, "b": y, **c }:
        assert_type(y, int)
        assert_type(c, dict[str, int])

y: dict[str, object] = {}
match y:
    case { "a": int() }:
        assert_type(y["a"], int)
"#,
);

testcase!(
    test_empty_loop,
    r#"
# These generate syntax that is illegal, but reachable with parser error recovery

for x in []:
pass  # E: Expected an indented block

while True:
pass  # E: Expected an indented block
"#,
);

testcase!(
    test_match_implicit_return,
    r#"
def test1(x: int) -> int:
    match x:
        case _:
            return 1
def test2(x: int) -> int:  # E: Function declared to return `int`, but one or more paths are missing an explicit `return`
    match x:
        case 1:
            return 1
"#,
);

testcase!(
    test_match_class_narrow,
    r#"
from typing import assert_type

class A:
    x: int
    y: str
    __match_args__ = ("x", "y")

class B:
    x: int
    y: str
    __match_args__ = ("x", "y")

class C:
    x: int
    y: str
    __match_args__ = ("x", "y")

def fun(x: A | B | C) -> None:
    match x:
        case A(1, "a"):
            assert_type(x, A)
    match x:
        case B(2, "b"):
            assert_type(x, B)
    match x:
        case B(3, "B") as y:
            assert_type(x, B)
            assert_type(y, B)
    match x:
        case A(1, "a") | B(2, "b"):
            assert_type(x, A | B)
"#,
);

testcase!(
    test_match_class,
    r#"
from typing import assert_type, assert_never

class Foo:
    x: int
    y: str
    __match_args__ = ("x", "y")

class Bar:
    x: int
    y: str

class Baz:
    x: int
    y: str
    __match_args__ = (1, 2)

def fun(foo: Foo, bar: Bar, baz: Baz) -> None:
    match foo:
        case Foo(1, "a"):
            pass
        case Foo(a, b):
            assert_type(a, int)
            assert_type(b, str)
        case _:
            assert_never(foo)
    match foo:
        case Foo(x = b, y = a):
            assert_type(a, str)
            assert_type(b, int)
        case _:
            assert_never(foo)
    match foo:
        case Foo(a, b, c):  # E: Cannot match positional sub-patterns in `Foo`\n  Index 2 out of range for `__match_args__`
            pass
        case _:
            assert_never(foo)
    match bar:
        case Bar(1):  # E: Object of class `Bar` has no attribute `__match_args__`
            pass
        case Bar(a):  # E: Object of class `Bar` has no attribute `__match_args__`
            pass
        case _:
            assert_never(bar)
    match bar:
        case Bar(x = a):
            assert_type(a, int)
        case _:
            assert_never(bar)
    match baz:
        case Baz(1):  # E: Expected literal string in `__match_args__`
            pass
        case _:
            assert_never(baz)  # E: Argument `Baz` is not assignable to parameter `arg` with type `Never`
"#,
);

testcase!(
    test_match_sequence_len,
    r#"
from typing import assert_type
def test(x: tuple[object] | tuple[object, object] | list[object]) -> None:
    match x:
        case [int()]:
            assert_type(x[0], int)
        case [a]:
            assert_type(x, tuple[object] | list[object])
        case [a, b]:
            assert_type(x, tuple[object, object] | list[object])
"#,
);

testcase!(
    test_match_sequence_len_starred,
    r#"
from typing import assert_type
def test(x: tuple[int, ...] | tuple[int, *tuple[int, ...], int] | tuple[int, int, int]) -> None:
    match x:
        case [first, second, third, *middle, last]:
            # tuple[int, int, int] is narrowed away because the case requires least 4 elements
            assert_type(x, tuple[int, ...] | tuple[int, *tuple[int, ...], int])
"#,
);

testcase!(
    bug = "we don't narrow attributes in a positional pattern",
    test_match_class_union,
    r#"
from typing import assert_type, assert_never, Literal

class Foo:
    x: int
    y: str
    __match_args__ = ("x", "y")

class Bar:
    x: str
    __match_args__ = ("x",)

def test(x: Foo | Bar) -> None:
    match x:
        case Foo(1, "a"):
            # we should narrow x.x and x.y to literals
            assert_type(x, Foo)
            assert_type(x.x, int)
            assert_type(x.y, str)
        case Foo(x = 1, y = ""):
            assert_type(x, Foo)
            assert_type(x.x, Literal[1])
            assert_type(x.y, Literal[""])
        case Bar("bar"):
            assert_type(x, Bar)
            assert_type(x.x, str)  # we want to narrow this to Literal["bar"]

def test_keyword_irrefutable(x: Foo | Bar) -> None:
    match x:
        case Foo(x = b, y = a):
            assert_type(x, Foo)
            assert_type(a, str)
            assert_type(b, int)
        case Bar(a) as b:
            assert_type(x, Bar)
            assert_type(b, Bar)
            assert_type(a, str)
            assert_type(b, Bar)
        case _:
            assert_never(x)

def test_positional(x: Foo | Bar) -> None:
    match x:
        case Foo(1, "a"):
            pass
        case Foo(a, b):
            assert_type(x, Foo)
            assert_type(a, int)
            assert_type(b, str)
"#,
);

testcase!(
    test_match_sequence_concrete,
    r#"
from typing import assert_type, Never

def foo(x: tuple[int, str, bool, int]) -> None:
    match x:
        case [bool(), b, c, d]:
            assert_type(x[0], bool)
            assert_type(b, str)
            assert_type(c, bool)
            assert_type(d, int)
        case [a, *rest]:
            assert_type(a, int)
            assert_type(rest, list[str | bool | int])
        case [a, *middle, b]:
            assert_type(a, int)
            assert_type(b, int)
            assert_type(middle, list[str | bool])
        case [a, b, c, d, e]:
            assert_type(x, Never)
        case [a, b, *middle, c, d]:
            assert_type(a, int)
            assert_type(b, str)
            assert_type(c, bool)
            assert_type(d, int)
            assert_type(middle, list[Never])
        case [*first, c, d]:
            assert_type(first, list[int | str])
            assert_type(c, bool)
            assert_type(d, int)
"#,
);

testcase!(
    test_match_sequence_unbounded,
    r#"
from typing import assert_type, Never

def foo(x: list[int]) -> None:
    match x:
        case []:
            pass
        case [a]:
            assert_type(a, int)
        case [a, b, c]:
            assert_type(a, int)
            assert_type(b, int)
            assert_type(c, int)
        case [a, *rest]:
            assert_type(a, int)
            assert_type(rest, list[int])
        case [a, *middle, b]:
            assert_type(a, int)
            assert_type(b, int)
            assert_type(middle, list[int])
        case [*first, a]:
            assert_type(first, list[int])
            assert_type(a, int)
        case [*all]:
            assert_type(all, list[int])
"#,
);

testcase!(
    test_match_or,
    r#"
from typing import assert_type

x: list[int] = [1, 2, 3]

match x:
    case [a] | a: # E: name capture `a` makes remaining patterns unreachable
        assert_type(a, list[int] | int)
    case [b] | _:  # E: alternative patterns bind different names
        assert_type(b, int)  # E: `b` may be uninitialized

match x:
    case _ | _:  # E: Only the last subpattern in MatchOr may be irrefutable
        pass
"#,
);

testcase!(
    test_crashing_match_sequence,
    r#"
match []:
    case [[1]]:
        pass
    case _:
        pass
"#,
);

testcase!(
    test_crashing_match_star,
    r#"
match []:
    case *x: # E: Parse error: Star pattern cannot be used here
        pass
    case *x | 1: # E: Parse error: Star pattern cannot be used here # E: alternative patterns bind different names
        pass
    case 1 | *x: # E: Parse error: Star pattern cannot be used here # E: alternative patterns bind different names
        pass
"#,
);

testcase!(
    test_match_narrow_generic,
    r#"
from typing import assert_type
class C:
    x: list[int] | None

    def test(self):
        x = self.x
        match x:
            case list():
                assert_type(x, list[int])

    def test2(self):
        match self.x:
            case list():
                assert_type(self.x, list[int])
"#,
);

testcase!(
    test_error_in_test_expr,
    r#"
def f(x: None):
    if x.nonsense:  # E: Object of class `NoneType` has no attribute `nonsense`
        pass
    while x['nonsense']:  # E: `None` is not subscriptable
        pass
    "#,
);

// Regression test for a crash
testcase!(
    test_ternary_and_or,
    r#"
def f(x: bool, y: int):
    return 0 if x else (y or 1)
    "#,
);

testcase!(
    test_if_which_exits,
    r#"
def foo(val: int | None, b: bool) -> int:
    if val is None:
        if b:
            return 1
        else:
            return 2
    return val
"#,
);

testcase!(
    test_shortcuit_or_after_flow,
    r#"
bar: str = "bar"

def func():
    foo: str | None = None

    for x in []:
        for y in []:
            pass

    baz: str = foo or bar
"#,
);

testcase!(
    test_export_not_in_flow,
    r#"
if 0.1:
    vari = "test"
    raise SystemExit
"#,
);

testcase!(
    test_assert_not_in_flow,
    r#"
from typing import assert_type, Literal
if 0.1:
    vari = "test"
    raise SystemExit
assert_type(vari, Literal["test"]) # E: `vari` is uninitialized
"#,
);

testcase!(
    test_assert_false_terminates_flow,
    r#"
def test1() -> int:
    assert False
def test2() -> int:  # E: Function declared to return `int` but is missing an explicit `return`
    assert True
    "#,
);

testcase!(
    test_if_defines_variable_in_one_side,
    r#"
from typing import assert_type, Literal
def condition() -> bool: ...
if condition():
    x = 1
else:
    pass
assert_type(x, Literal[1])  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_while_true_defines_variable,
    r#"
from typing import assert_type, Literal
def foo():
    while True:
        x = "a"
        break
    assert_type(x, Literal["a"])
    "#,
);

testcase!(
    test_while_true_redefines_and_narrows_variable,
    r#"
from typing import assert_type, Literal
def get_new_y() -> int | None: ...
def foo():
    y = None
    while True:
        if (y := get_new_y()):
            break
    assert_type(y, int)
    "#,
);

testcase!(
    test_nested_if_sometimes_defines_variable,
    r#"
from typing import assert_type, Literal
def condition() -> bool: ...
if condition():
    if condition():
        x = "x"
else:
    x = "x"
print(x)  # E: `x` may be uninitialized
    "#,
);

testcase!(
    test_named_inside_boolean_op,
    r#"
from typing import assert_type, Literal
b: bool = True
y = 5
x0 = True or (y := b) and False
assert_type(y, Literal[5] | bool)  # this is as expected
x0 = True or (z := b) and False
# This is an intended false negative uninitialized local check: because we can't
# distinguish different downstream uses fully, we disable uninitialized local
# checks for names defined in bool ops.
assert_type(z, bool)
"#,
);

testcase!(
    test_redundant_condition_func,
    r#"
def foo() -> bool: ...

if foo:  # E: Function object `foo` used as condition
    ...
while foo:  # E: Function object `foo` used as condition
    ...
[x for x in range(42) if foo]  # E: Function object `foo` used as condition
    "#,
);

testcase!(
    test_redundant_condition_class,
    r#"
class Foo:
    def __bool__(self) -> bool: ...

if Foo:  # E: Class name `Foo` used as condition
    ...
while Foo:  # E: Class name `Foo` used as condition
    ...
[x for x in range(42) if Foo]  # E: Class name `Foo` used as condition
    "#,
);

testcase!(
    test_redundant_condition_int,
    r#"
if 42:  # E: Integer literal used as condition. It's equivalent to `True`
    ...
while 0:  # E: Integer literal used as condition. It's equivalent to `False`
    ...
[x for x in range(42) if 42]  # E: Integer literal used as condition
    "#,
);

testcase!(
    test_redundant_condition_str_bytes,
    r#"
if "test":  # E: String literal used as condition. It's equivalent to `True`
    ...
while "":  # E: String literal used as condition. It's equivalent to `False`
    ...
[x for x in range(42) if b"test"]  # E: Bytes literal used as condition
    "#,
);

testcase!(
    test_redundant_condition_enum,
    r#"
import enum
class E(enum.Enum):
    A = 1
    B = 2
    C = 3
if E.A:  # E: Enum literal `E.A` used as condition
    ...
while E.B:  # E: Enum literal `E.B` used as condition
    ...
[x for x in range(42) if E.C]  # E: Enum literal `E.C` used as condition
    "#,
);

testcase!(
    crash_no_try_type,
    r#"
# Used to crash, https://github.com/facebook/pyrefly/issues/766
try:
    pass
except as r: # E: Parse error: Expected one or more exception types
    pass
"#,
);

testcase!(
    test_narrows_in_flow_merge_when_not_in_base_flow,
    r#"
from typing import reveal_type
class A: pass
class B(A): pass
class C(A): pass
x: A = A()
y: A = A()
def f():
    if isinstance(x, B):
        assert isinstance(y, B)
        pass
    elif isinstance(x, C):
        assert isinstance(y, C)
        pass
    reveal_type(x)  # E: revealed type: A
    reveal_type(y)  # E: revealed type: A
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/77
testcase!(
    loop_with_sized_operation,
    r#"
intList: list[int] = [5, 6, 7, 8]
for j in [1, 2, 3, 4]:
    for i in range(len(intList)):
        intList[i] *= 42
print([value for value in intList])
"#,
);

testcase!(
    bug = "For now, we disabled uninitialized local check for walrus in bool op, see #1251",
    test_walrus_names_in_bool_op_straight_line,
    r#"
def condition() -> bool: ...
def f_and():
    b = (z := condition()) and (y := condition())
    print(z)
    print(y)  # Intended false negative
def f_or():
    b = (z := condition()) or (y := condition())
    print(z)
    print(y)  # Intended false negative

    "#,
);

testcase!(
    bug = "For now, we disabled uninitialized local check for walrus in bool op, see #1251",
    test_walrus_names_in_bool_op_as_guard,
    r#"
def condition() -> bool: ...
def f_and():
    if (z := condition()) or (y := condition()):
        print(z)
        print(y)  # Intended false negative
def f_or():
    if (z := condition()) and (y := condition()):
        print(z)
        print(y)  # Note this is *not* a false negative
    "#,
);

testcase!(
    test_setitem_with_loop_and_walrus,
    r#"
def f():
    d: dict[int, int] = {}
    for i in range(10):
        idx = i
        d[idx] = (x := idx)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/528
testcase!(
    test_narrow_in_branch_contained_in_loop,
    r#"
from typing import Iterable, Iterator, cast

def iterate[T](*items: T | Iterable[T]) -> Iterator[T]:
    for item in items:
        if isinstance(item, str):
            yield cast(T, item)
        elif isinstance(item, Iterable):
            yield from item
        else:
            yield item
"#,
);

testcase!(
    test_bad_setitem_with_loop_and_walrus,
    r#"
def f():
    d: dict[str, int] = {}
    for i in range(10):
        idx = i
        d[idx] = (x := idx)  # E: `int` is not assignable to parameter `key` with type `str`
    "#,
);

testcase!(
    test_walrus_on_first_branch_of_if,
    r#"
def condition() -> bool: ...
def f1() -> bool:
    if (b := condition()):
        pass
    return b

def f2() -> bool:
    if (a := condition()) and condition() and (b := condition()):
        return b
    return a

def f3() -> bool:
    if (a := condition()) and (b := condition()):
        return b
    return a
    "#,
);

// When a variable is PossiblyUninitialized (defined in only one branch of an
// if/else) and then redefined via walrus in a BoolOp, the PossiblyUninitialized
// from the short-circuit branch poisons the BoolOp merge via the early return in
// FlowStyle::merged (line that matches PossiblyUninitialized), bypassing the
// BoolOp laxness. Inside the if-body all `and` operands succeeded, so the walrus
// must have executed and the variable is definitely initialized.
testcase!(
    test_walrus_in_boolop_after_possibly_uninitialized,
    r#"
def condition() -> bool: ...
def get() -> int: ...

def f1(x: bool) -> None:
    if x:
        viewer = 1
    # viewer is PossiblyUninitialized here
    if condition() and (viewer := get()):
        print(viewer)

def f2(x: bool) -> None:
    """Same pattern but the first definition is in the else branch."""
    if x:
        pass
    else:
        if condition() and (viewer := get()):
            pass
    # viewer is PossiblyUninitialized here
    if condition() and (viewer := get()):
        print(viewer)

def f3(x: bool) -> None:
    """Three-way and chain, matching the real-world pattern."""
    if x:
        pass
    else:
        viewer = 1
    if get() and get() and (viewer := get()):
        print(viewer)
    "#,
);

// Short-circuit prevents `value := v` from executing when the lhs is `False`.
// However, processing the test before the fork applies BoolOp lax semantics, so
// `value` appears maybe-initialized — a known false negative from BoolOp laxness.
// This is the same trade-off as test_walrus_names_in_bool_op_straight_line.
testcase!(
    bug = "BoolOp laxness causes false negative for walrus in short-circuit context, see #1251",
    test_false_and_walrus,
    r#"
def f(v):
    if False and (value := v):
        print(value)
    else:
        print(value)
    "#,
);

// Regression tests for https://github.com/facebook/pyrefly/issues/2382
// Walrus operator in ternary test expression

testcase!(
    test_walrus_in_ternary_else_branch,
    r#"
def f(i: float) -> int:
    return a if (a := round(i)) - 1 else a + 1
    "#,
);

testcase!(
    test_walrus_in_ternary_only_in_else,
    r#"
def f(x: int) -> int:
    return 0 if (y := x) > 0 else y
    "#,
);

// x is narrowed to int in the body (is not None) and
// the else branch returns 0 (int), so the return type is int. No error.
testcase!(
    test_walrus_in_ternary_with_narrowing,
    r#"
from typing import assert_type
def get() -> int | None: ...
def f() -> int:
    return x if (x := get()) is not None else 0
    "#,
);

testcase!(
    test_walrus_ternary_truthiness_narrowing,
    r#"
from typing import assert_type
def get() -> str | None: ...
def f() -> str:
    return x if (x := get()) else "default"
    "#,
);

testcase!(
    test_walrus_in_ternary_short_circuit,
    r#"
def condition() -> bool: ...
def get() -> int: ...
def f1() -> int:
    return x if condition() and (x := get()) else 0  # no error
# BoolOp merging uses lax handling, so `x` is treated as defined even though
# `x := get()` may not execute. This is a known false negative from BoolOp laxness.
def f2() -> int:
    return x if condition() or (x := get()) else 0  # false negative
def f3() -> int:
    return x if condition() and (x := get()) else x  # false negative
    "#,
);

// Walrus in outer ternary test: `a` should be visible in both branches.
// Currently this works because truthiness narrowing on `a` adds it to the
// else flow, masking the uninitialized status.
testcase!(
    test_walrus_in_nested_ternary_outer,
    r#"
def f(v: int) -> int:
    return (a if a > 0 else -a) if (a := v) else -a
    "#,
);

testcase!(
    test_walrus_in_nested_ternary_inner,
    r#"
def condition() -> bool: ...
def get() -> int: ...
def f() -> int:
    return (b if (b := get()) > 0 else 0) if condition() else -1
    "#,
);

// Regression tests for https://github.com/facebook/pyrefly/issues/2382
// Walrus operator in if-statement test conditions

// The first `if` test always evaluates, so walrus bindings should be in base flow.
testcase!(
    test_walrus_in_if_basic,
    r#"
def f(a: int) -> int:
    if (x := a) > 0:
        pass
    return x
    "#,
);

testcase!(
    test_walrus_in_if_both_branches,
    r#"
def f(a: int) -> int:
    if (x := a) > 0:
        result = x + 1
    else:
        result = x - 1
    return result
    "#,
);

testcase!(
    test_walrus_in_if_with_narrowing,
    r#"
def get() -> int | None: ...
def f() -> int:
    if (x := get()) is not None:
        return x
    return 0
    "#,
);

// elif condition only executes if the first `if` was False — walrus may not run.
testcase!(
    bug = "In order to fix false positives, we handled narrows differently in if/elif and introduced a false negative here",
    test_walrus_in_elif,
    r#"
def condition() -> bool: ...
def f() -> bool:
    if condition():
        pass
    elif (x := condition()):
        pass
    return x  # False negative: x winds up getting applied as if it were in the base flow due to the negative narrow
    "#,
);

// When the `if` branch raises, the elif condition must execute before
// reaching code after the if/elif block, so the walrus is always assigned.
testcase!(
    test_walrus_in_elif_with_raise,
    r#"
def foo() -> bool:
    return True

def bar() -> int:
    return 1

def f() -> None:
    if not foo():
        raise AssertionError()
    elif (_bar := bar()) > 1:
        raise AssertionError()
    print(_bar)
    "#,
);

// The walrus assignment should propagate to the base flow so the
// merge does not falsely report "may be uninitialized".
testcase!(
    test_walrus_in_elif_targeting_declared_local,
    r#"
def foo() -> bool:
    return True

def bar() -> int:
    return 1

def f() -> None:
    x: int
    if not foo():
        raise AssertionError()
    elif (x := bar()) > 1:
        raise AssertionError()
    print(x)
    "#,
);

testcase!(
    test_walrus_multiple_elif,
    r#"
def foo() -> bool:
    return True

def bar() -> int:
    return 1

def f() -> None:
    if not foo():
        raise AssertionError()
    elif (x := bar()) > 1:
        raise AssertionError()
    elif (y := bar()) > 2:
        raise AssertionError()
    print(x)
    print(y)
    "#,
);

testcase!(
    test_walrus_in_elif_with_else,
    r#"
def foo() -> bool:
    return True

def bar() -> int:
    return 1

def f() -> None:
    if not foo():
        raise AssertionError()
    elif (x := bar()) > 1:
        pass
    else:
        pass
    print(x)
    "#,
);

// the walrus may not execute, so x should be possibly-uninitialized.
testcase!(
    bug = "Should report x as possibly uninitialized since the if branch does not terminate",
    test_walrus_in_elif_preceding_if_no_terminate,
    r#"
def foo() -> bool:
    return True

def bar() -> int:
    return 1

def f() -> None:
    if foo():
        pass
    elif (x := bar()) > 1:
        pass
    print(x)  # should be an error: x may be uninitialized
    "#,
);

testcase!(
    test_walrus_in_if_no_else,
    r#"
def f(a: int) -> int:
    if (x := a) > 0:
        return x
    return x
    "#,
);

testcase!(
    test_trycatch_implicit_return,
    r#"
def f() -> int:
    try:
        return 1
    finally:
        pass
    "#,
);

testcase!(
    test_merging_any,
    r#"
from typing import Any, assert_type
def f(x: Any, y: Any):
    if isinstance(x, int):
        y = "y"
    assert_type(x, Any)
    assert_type(y, Any)
    "#,
);

testcase!(
    test_reducible_join_of_narrows,
    r#"
from typing import assert_type
class A: pass
class B(A): pass
def f(x: A):
    if isinstance(x, B):
        pass
    assert_type(x, A)
    "#,
);

testcase!(
    test_join_with_unrelated_narrow,
    r#"
from typing import assert_type, reveal_type
class A: pass
class B: pass
def f(x: A):
    if isinstance(x, B):
        reveal_type(x) # E: A & B
    assert_type(x, A)
# (Illustrating that all code in the body of `f` is reachable)
class C(A, B): pass
f(C())
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1246
testcase!(
    test_boolean_op_narrowing_example,
    r#"
from typing import Sequence, assert_type
class A:
    def foo(self) -> bool:
        raise NotImplementedError()
    def i(self) -> int:
        raise NotImplementedError()
class B:
    def bar(self) -> bool:
        raise NotImplementedError()
def f(a: A) -> tuple[int, bool]:
    return a.i(), (
        (isinstance(a, B) and a.bar()) or
        assert_type(a, A).foo()
    )
"#,
);

fn env_pytest() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "pytest",
        "pytest.pyi",
        r#"
from typing import NoReturn
def fail(x: str) -> NoReturn: ...
"#,
    );
    t
}

testcase!(
    test_pytest_noreturn,
    env_pytest(),
    r#"
import pytest

def test_oops() -> None:
    try:
        val = True
    except:
        pytest.fail("execution stops here")
    assert val, "oops"
"#,
);

#[test]
fn test_many_subscript_assignments_do_not_stack_overflow() -> anyhow::Result<()> {
    let mut contents =
        String::from("new_val: dict[str, int] = {}\nvalues: dict[str, dict[str, int]] = {}\n");
    for i in 0..600 {
        writeln!(&mut contents, "values[\"K{i}\"] = {{}}").unwrap();
    }
    for i in 0..600 {
        writeln!(&mut contents, "values[\"K{i}\"][\"K{i}\"] = {i}").unwrap();
    }
    testcase_for_macro(TestEnv::new(), &contents, file!(), line!())
}

// Regression test for a stack overflow we had at one point.
testcase!(
    test_flow_merging_with_recursion,
    r#"
def test(xs: list[int], ys: list[int], zs: list[int]) -> None:
    results = []
    for _ in xs:
        for _ in ys:
            for _ in zs:
                if len(results) >= 0:
                    break
            if True and len(results) >= 0:
                break
"#,
);

// These types (bool, bytearray, bytes, dict, float, frozenset, int, list, set, str, tuple)
// bind the entire narrowed value to the single positional parameter instead of using __match_args__
testcase!(
    test_pattern_match_single_slot_builtins,
    r#"
from typing import assert_type

def test_float(x: object) -> None:
    match x:
        case float(value):
            assert_type(value, float)

def test_int(x: object) -> None:
    match x:
        case int(value):
            assert_type(value, int)

def test_str(x: object) -> None:
    match x:
        case str(value):
            assert_type(value, str)

def test_bool(x: object) -> None:
    match x:
        case bool(value):
            assert_type(value, bool)

def test_bytes(x: object) -> None:
    match x:
        case bytes(value):
            assert_type(value, bytes)

def test_bytearray(x: object) -> None:
    match x:
        case bytearray(value):
            assert_type(value, bytearray)

def test_list(x: list[int]) -> None:
    match x:
        case list(value):
            assert_type(value, list[int])

def test_tuple(x: tuple[int, str]) -> None:
    match x:
        case tuple(value):
            assert_type(value, tuple[int, str])

def test_set(x: set[int]) -> None:
    match x:
        case set(value):
            assert_type(value, set[int])

def test_frozenset(x: frozenset[int]) -> None:
    match x:
        case frozenset(value):
            assert_type(value, frozenset[int])

def test_dict(x: dict[str, int]) -> None:
    match x:
        case dict(value):
            assert_type(value, dict[str, int])

def test_narrowing_with_union(x: int | str) -> None:
    match x:
        case int(value):
            assert_type(value, int)
            assert_type(x, int)
        case str(value):
            assert_type(value, str)
            assert_type(x, str)

# Test that multiple positional patterns error
def test_multiple_positional_not_special(x: int) -> None:
    match x:
        case int(a, b):  # E: Cannot match positional sub-patterns in `int` # E: Object of class `int` has no attribute `__match_args__`
            pass

# Test that keyword patterns still work normally
def test_keyword_pattern_not_special(x: float) -> None:
    match x:
        case float(real=r):
            assert_type(r, float)
"#,
);

testcase!(
    test_pep765_break_continue_return_in_finally_3_14,
    TestEnv::new_with_version(PythonVersion {
        major: 3,
        minor: 14,
        micro: 0,
    }),
    r#"
def test():
    try:
        pass
    finally:
        return # E: in a `finally` block

for _ in []:
    try:
        pass
    finally:
        break # E: in a `finally` block
for _ in []:
    try:
        pass
    finally:
        continue # E: in a `finally` block

try:
    pass
finally:
    def f():
        return 42 # OK

try:
    pass
finally:
    def f():
        try:
            pass
        finally:
            return 42 # E: in a `finally` block

try:
    pass
finally:
    for _ in []:
        try:
            pass
        finally:
            break # E: in a `finally` block
    "#,
);

testcase!(
    test_pep765_break_continue_return_in_finally_3_13,
    TestEnv::new_with_version(PythonVersion {
        major: 3,
        minor: 13,
        micro: 0,
    }),
    r#"
# For now, we won't emit a PEP765 syntax error for 3.13 and below
def test():
    try:
        pass
    finally:
        return

for _ in []:
    try:
        pass
    finally:
        break
for _ in []:
    try:
        pass
    finally:
        continue
    "#,
);

testcase!(
    test_noreturn_branch_termination,
    r#"
from typing import NoReturn, assert_type

def raises() -> NoReturn:
    raise Exception()

def f(x: str | bytes | bool) -> str | bytes:
    if isinstance(x, str):
        pass
    elif isinstance(x, bytes):
        pass
    else:
        raises()
    return x  # Should be ok - x is str | bytes here

def g(x: str | None) -> str:
    if x is None:
        raises()
    return x  # Should be ok - x is str here

def h(x: int | str) -> None:
    if isinstance(x, int):
        y = x + 1
    else:
        raises()
    assert_type(y, int)  # y should be int, not str | int
"#,
);

testcase!(
    test_noreturn_nested_branches,
    r#"
from typing import NoReturn, assert_type

def raises() -> NoReturn:
    raise Exception()

def f(x: str | int | None) -> str:
    if x is None:
        raises()
    else:
        if isinstance(x, str):
            return x
        else:
            raises()
    # Should not be reachable, but if it were, x would be str
"#,
);

testcase!(
    test_noreturn_with_assignment_after,
    r#"
from typing import assert_type, NoReturn

def raises() -> NoReturn:
    raise Exception()

def f(x: str | None):
    if x is None:
        raises()
        y = "unreachable"  # This makes the branch NOT terminate
    assert_type(x, str | None)
"#,
);

testcase!(
    test_noreturn_all_branches_terminate,
    r#"
from typing import assert_type, NoReturn, Never

def raises() -> NoReturn:
    raise Exception()

def f(x: int | str):
    if isinstance(x, str):
        raises()
    else:
        raises()
    assert_type(x, Never)
"#,
);

testcase!(
    test_non_noreturn_with_termination_key,
    r#"
from typing import assert_type

def maybe_raises() -> None:
    """Not NoReturn - might return normally."""
    if True:
        raise Exception()

def f(cond: bool) -> str:
    if cond:
        x = "defined"
    else:
        maybe_raises()  # Has termination key, but is NOT NoReturn
    return x  # E: `x` may be uninitialized
"#,
);

testcase!(
    test_non_noreturn_elif,
    r#"
def maybe_raises() -> None:
    if True:
        raise Exception()

def f(x: int) -> str:
    if x == 1:
        y = "one"
    elif x == 2:
        maybe_raises()
    else:
        maybe_raises()
    return y  # E: `y` may be uninitialized
"#,
);

testcase!(
    test_declared_variable_with_noreturn_else_false_positive,
    r#"
from typing import NoReturn

def raises() -> NoReturn:
    raise Exception()

def f(x: int) -> str:
    y: str
    if x == 1:
        y = "one"
    elif x == 2:
        y = "two"
    else:
        raises()
    return y
"#,
);

testcase!(
    test_if_elif_enum_exhaustive,
    r#"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def f(c: Color) -> str:
    if c == Color.RED:
        return "warm"
    elif c == Color.GREEN:
        return "natural"
    elif c == Color.BLUE:
        return "cool"
"#,
);

testcase!(
    test_if_elif_isinstance_exhaustive,
    r#"
def f(x: int | str) -> str:
    if isinstance(x, int):
        return "int"
    elif isinstance(x, str):
        return "str"
"#,
);

testcase!(
    test_if_elif_non_exhaustive,
    r#"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def f(c: Color) -> str:  # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    if c == Color.RED:
        return "warm"
    elif c == Color.GREEN:
        return "natural"
    # Missing Color.BLUE case - should always error
"#,
);

testcase!(
    test_if_elif_with_else_trivially_exhaustive,
    r#"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def f(c: Color) -> str:
    if c == Color.RED:
        return "warm"
    elif c == Color.GREEN:
        return "natural"
    else:
        return "cool"
"#,
);

testcase!(
    test_if_elif_literal_union_exhaustive,
    r#"
from typing import Literal

def f(x: Literal["a", "b", "c"]) -> str:
    if x == "a":
        return "first"
    elif x == "b":
        return "second"
    elif x == "c":
        return "third"
"#,
);

testcase!(
    test_if_elif_mixed_narrowing,
    r#"
def f(x: int | None) -> str:
    if x is None:
        return "none"
    elif isinstance(x, int):
        return "int"
"#,
);

testcase!(
    test_if_elif_bool_exhaustive,
    r#"
def f(x: bool) -> str:
    if x:
        return "true"
    elif not x:
        return "false"
"#,
);

testcase!(
    test_if_elif_multiple_subjects,
    r#"
def f(x: int | str, y: int | str) -> str:  # E: Function declared to return `str`, but one or more paths are missing an explicit `return`
    if isinstance(x, int):
        return "x is int"
    elif isinstance(y, str):
        return "y is str"
    # Different subjects in different branches - cannot determine exhaustiveness
"#,
);

testcase!(
    test_if_elif_mixed_subjects_one_exhaustive,
    r#"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def f(x: Color, y: int | str) -> str:
    if x == Color.RED:
        return "red"
    elif isinstance(y, int):
        return "y is int"
    elif x == Color.GREEN:
        return "green"
    elif x == Color.BLUE:
        return "blue"
"#,
);

// Regression test for the first example bug reported in https://github.com/facebook/pyrefly/issues/1286
testcase!(
    test_match_can_narrow_union_to_never_in_wildcard,
    r#"
from typing import assert_never
class A:...
class B:...

def go(mdl:A|B):
    match mdl:
        case A():
            print('A')
        case B():
            print('B')
        case _:
            assert_never(mdl)
    "#,
);

testcase!(
    test_match_keyword_wildcard_pattern_is_irrefutable,
    r#"
from dataclasses import dataclass
from typing import assert_never

@dataclass
class A: ...

@dataclass
class B:
    x: int

T = A | B

def test(x: T):
    match x:
        case A(): ...
        case B(x=_): ...
        case _:
            assert_never(x)
    "#,
);

testcase!(
    test_match_exhausts_literal_type,
    r#"
from typing import Literal, assert_never

type A = Literal['A']

class C:
    def __init__(self, a: A) -> None:
        self.a = a

    def f(self) -> None:
        match self.a:
            case 'A':
                pass
            case ever:
                assert_never(ever)
    "#,
);

// Regression test for the third example bug reported in https://github.com/facebook/pyrefly/issues/1286
testcase!(
    test_enum_exhaustive_match_and_uninitialized_local,
    r#"
from enum import IntEnum

class Rating(IntEnum):
    Again = 1
    Hard = 2
    Good = 3
    Easy = 4

def foo()->Rating:
    ...

x = foo()
match x:
    case Rating.Again:
        y = 1
    case Rating.Easy | Rating.Good | Rating.Hard:
        y = 2
print(y)
    "#,
);

// Issue #2406: NoReturn in except block should make variable always initialized
testcase!(
    test_noreturn_try_except_simple,
    r#"
from typing import NoReturn

def foo() -> NoReturn:
    raise ValueError('')

def main() -> None:
    try:
        node = 1
    except Exception:
        foo()
    print(node)
"#,
);

testcase!(
    test_noreturn_try_except_if_nested,
    r#"
from typing import NoReturn

def foo() -> NoReturn:
    raise ValueError('')

def main(resolve: bool) -> None:
    try:
        node = 1
    except Exception as exc:
        foo()
    if resolve:
        try:
            node = 2
        except Exception:
            foo()
    print(node)
"#,
);

// for https://github.com/facebook/pyrefly/issues/1840
testcase!(
    test_exhaustive_flow_no_fall_through,
    r#"
import types
from dataclasses import dataclass
from typing import Any, TypeIs, assert_never


def is_instance_union_aware[T](
    value: Any, target_type: type[T] | tuple[type[T], ...]
) -> TypeIs[T]: ...

def test_is_instance_union_aware():
    @dataclass
    class C0:
        f_common: int
        f_0: int

    @dataclass
    class C1:
        f_common: int
        f_1: int

    @dataclass
    class C2:
        f_common: int
        f_2: int

    def compute_1(obj: C0 | C1 | C2) -> int:
        if is_instance_union_aware(obj, C0 | C1):
            return obj.f_common
        return obj.f_2 + obj.f_common

    def compute_2(obj: C0 | C1 | C2) -> int:
        if is_instance_union_aware(obj, C0 | C1):
            return obj.f_common
        if is_instance_union_aware(obj, C2):
            return obj.f_2 + obj.f_common
        assert_never(obj)

    assert compute_1(C1(f_common=1, f_1=2)) == 3
    assert compute_2(C1(f_common=4, f_1=5)) == 9
    "#,
);

// https://github.com/facebook/pyrefly/issues/1896
testcase!(
    test_exhaustive_flow_no_early_return_narrow,
    r#"
import dataclasses as dc
from typing import assert_type

@dc.dataclass(frozen=True)
class Success:
    value: int

@dc.dataclass(frozen=True)
class Error:
    message: str

Result = Success | Error | None

def get_result() -> Result:
    return Success(value=42)

def use_success(s: Success) -> int:
    return s.value

def demo_pyre_narrowing_failure() -> int:
    result = get_result()
    match result:
        case Error() as err:
            return -1
        case None:
            return 0
        case _:
            success = result
    assert_type(success, Success)
    return use_success(success)
    "#,
);

// https://github.com/facebook/pyrefly/issues/2261
testcase!(
    test_walrus_in_if_with_is_none,
    r#"
def fun(**kwargs):
    if x := kwargs.get("x") is None:
        x = "a"
    print(x)
    "#,
);

// https://github.com/facebook/pyrefly/issues/1397
testcase!(
    test_walrus_in_chained_if_re_match,
    r#"
from re import compile

interface_re = compile(r"^foo")
ipv4_re = compile(r"bar$")
line = str()

if match := interface_re.match(line):
    pass

if line and (match := ipv4_re.search(line)):
    print(match)
    "#,
);

// https://github.com/facebook/pyrefly/issues/1397
testcase!(
    test_walrus_in_negated_if_with_isinstance,
    r#"
from typing import Any

def test(thing: Any) -> None:
    if not (items := getattr(thing, "items")):
        return
    if not isinstance(items, tuple|list):
        items = (items,)
    for item in items:
        print(item)
    "#,
);

// https://github.com/facebook/pyrefly/issues/1397
testcase!(
    test_walrus_bool_in_if,
    r#"
def f() -> None:
    if a := True:
        print(a)
    print(a)
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_in_method_call_chain,
    r#"
import pathlib

def f(mod: str, stubs_path: pathlib.Path):
    _, *submods = mod.split(".")
    if (path := stubs_path.joinpath(*submods, "__init__.pyi")).is_file():
        return path
    assert submods, path
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_in_comparison,
    r#"
def check():
    if (y := 2) <= 1:
        return
    print(y)
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_with_and_condition,
    r#"
def f(v):
    x: int
    if (x := v) and v:
        print(x)
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_in_compound_and_condition,
    r#"
def hello(x: int, y: int) -> int | None:
    if x == 5 and (z := x + y) == 7:
        return z
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_with_none_reassignment,
    r#"
d: dict[str, str] = {}
def func(key: str) -> str:
    if (name := d.get(key)) is None:
        name = 'missing'
    d[key] = name
    return name
    "#,
);

// https://github.com/facebook/pyrefly/issues/913
testcase!(
    test_walrus_in_loop_with_narrowing,
    r#"
from typing import assert_type
d1 = {0: '0', 1:'1', 3:'3'}
d2 = {'0': 0, '1': 1, '2': 2, '3':3}
for x in range(10):
    if not (y := d1.get(x)):
        continue
    assert_type(y, str)
    if (z := d2[y]) < 2:
        assert_type(z, int)
        continue
    assert_type(z, int)
    "#,
);

// When a variable is defined inside `if a:` and used inside a subsequent
// `if a:`, the variable is guaranteed to be initialized because the same
// condition guards both the definition and the use.
testcase!(
    bug = "false positive: b is always initialized when a is truthy",
    test_guarded_initialization_basic,
    r#"
def f(a: bool) -> int:
    if a:
        b = 3
    c = 5
    if a:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    test_guarded_initialization_negated_condition,
    r#"
def f(a: bool) -> int:
    if a:
        b = 3
    if not a:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    test_guarded_initialization_unrelated_condition,
    r#"
def f(a: bool, c: bool) -> int:
    if a:
        b = 3
    if c:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    bug = "false positive: b and c are always initialized when a is truthy",
    test_guarded_initialization_multiple_variables,
    r#"
def f(a: bool) -> int:
    if a:
        b = 3
        c = 4
    if a:
        return b + c  # E: `b` may be uninitialized  # E: `c` may be uninitialized
    return 0
    "#,
);

testcase!(
    bug = "false positive: b is always initialized when a is truthy",
    test_guarded_initialization_with_intermediate_statements,
    r#"
def f(a: bool) -> int:
    if a:
        b = 3
    x = 5
    y = x + 1
    if a:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    bug = "false positive: b is always initialized when a is truthy",
    test_guarded_initialization_annotation_then_guarded_assign,
    r#"
def f(a: bool) -> int:
    b: int
    if a:
        b = 3
    if a:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    bug = "false positive: b is always initialized when a is truthy",
    test_guarded_initialization_repeated_use,
    r#"
def f(a: bool) -> None:
    if a:
        b = 3
    if a:
        print(b)  # E: `b` may be uninitialized
    if a:
        print(b)
    "#,
);

testcase!(
    test_guarded_initialization_guard_reassigned,
    r#"
def f(a: bool, c: bool) -> int:
    if a:
        b = 3
    a = c
    if a:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);

testcase!(
    bug = "false positive: b is always initialized when a > 0 at both sites",
    test_guarded_initialization_complex_condition,
    r#"
def f(a: int) -> int:
    if a > 0:
        b = 3
    if a > 0:
        return b  # E: `b` may be uninitialized
    return 9
    "#,
);
