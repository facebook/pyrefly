/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
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
break  # E: Cannot `break` outside loop
continue  # E: Cannot `continue` outside loop
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
from typing import reveal_type

class Exception1(Exception): pass
class Exception2(Exception): pass

x1: tuple[type[Exception], ...] = (Exception1, Exception2)
x2 = (Exception1, Exception2)

try:
    pass
except int as e1:  # E: Invalid exception class: `int` does not inherit from `BaseException`
    reveal_type(e1)  # E: revealed type: int
except int:  # E: Invalid exception class
    pass
except Exception as e2:
    reveal_type(e2)  # E: revealed type: Exception
except ExceptionGroup as e3:
    reveal_type(e3)  # E: revealed type: ExceptionGroup[Exception]
except (Exception1, Exception2) as e4:
    reveal_type(e4)  # E: revealed type: Exception1 | Exception2
except Exception1 as e5:
    reveal_type(e5)  # E: revealed type: Exception1
except x1 as e6:
    reveal_type(e6)  # E: revealed type: Exception
except x2 as e7:
    reveal_type(e7)  # E: revealed type: Exception1 | Exception2
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
    reveal_type(e2)  # E: revealed type: ExceptionGroup[Exception]
except* ExceptionGroup as e3:  # E: Exception handler annotation in `except*` clause may not extend `BaseExceptionGroup`
    reveal_type(e3)  # E: ExceptionGroup[ExceptionGroup[Exception]]
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
            assert_type(x, int)
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
            assert_type(x, A | B | C)
            assert_type(y, B)
    match x:
        case A(1, "a") | B(2, "b"):
            assert_type(x, A | B)
"#,
);

testcase!(
    test_match_class,
    r#"
from typing import assert_type

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
        case Foo(x = b, y = a):
            assert_type(a, str)
            assert_type(b, int)
        case Foo(a, b, c):  # E: Cannot match positional sub-patterns in `Foo`\n  Index 2 out of range for `__match_args__`
            pass
    match bar:
        case Bar(1):  # E: Object of class `Bar` has no attribute `__match_args__`
            pass
        case Bar(a):  # E: Object of class `Bar` has no attribute `__match_args__`
            pass
        case Bar(x = a):
            assert_type(a, int)
    match baz:
        case Baz(1):  # E: Expected literal string in `__match_args__`
            pass
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
from typing import assert_type, Literal

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
        case Foo(a, b):
            assert_type(x, Foo)
            assert_type(a, int)
            assert_type(b, str)
        case Foo(x = b, y = a):
            assert_type(x, Foo)
            assert_type(a, str)
            assert_type(b, int)
        case Foo(x = 1, y = ""):
            assert_type(x, Foo)
            assert_type(x.x, Literal[1])
            assert_type(x.y, Literal[""])
        case Bar("bar"):
            assert_type(x, Bar)
            assert_type(x.x, str)  # we want to narrow this to Literal["bar"]
        case Bar(a) as b:
            assert_type(x, Foo | Bar)
            assert_type(b, Bar)
            assert_type(a, str)
            assert_type(b, Bar)
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
    case [b] | _:
        assert_type(b, int)  # E: `b` may be uninitialized

match x:
    case _ | _:  # E: Only the last subpattern in MatchOr may be irrefutable
        pass
"#,
);

testcase!(
    test_crashing_match,
    r#"
match []:
    case [[1]]:
        pass
    case _:
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
    bug = "We unsoundly merge narrows dropping missing flow. See the incorrect reveal_type(y) result.",
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
    # We get this case right, but for a brittle reason: we negate the tests in the base flow.
    reveal_type(x)  # E: revealed type: A | B | C
    # The negation trick doesn't work here, and we get an incorrect narrow.
    reveal_type(y)  # E: revealed type: B | C
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
    bug = "We approximate flow for tests in a lossy way - the first test actually runs in the base flow",
    test_walrus_on_first_branch_of_if,
    r#"
def condition() -> bool: ...
def f() -> bool:
    if (b := condition()):
        pass
    # In our approximation, `b` is defined in the branch but actually the test always evaluates
    return b  # E: `b` may be uninitialized
    "#,
);

testcase!(
    test_false_and_walrus,
    r#"
def f(v):
    if False and (value := v):
        print(value)
    else:
        print(value)  # E: `value` is uninitialized
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

// Regression test for https://github.com/facebook/pyrefly/issues/683
testcase!(
    test_loop_with_sized_in_inner_iteration,
    r#"
def f(xs: list[list]):
    for x in xs:
        for i in range(len(x)):
            x[i] = 1
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/812
testcase!(
    test_loop_with_set_and_len,
    r#"
def f(my_set: set[int]):
    while True:
        start_size = len(my_set)
        my_set.update([])
        if len(my_set) == start_size:
            return
"#,
);

// Regression test: at one point, excessive loop recursion caused the reveal type to be `Unknown`
testcase!(
    test_loop_with_dict_get,
    r#"
from typing import reveal_type
def f(keys: list[str]):
    counters: dict[str, int] = {}
    for k in keys:
        counters[k] = reveal_type(counters.get(k, 0))  # E: revealed type: int
"#,
);

testcase!(
    test_while_simple,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = None
    while condition():
        assert_type(x, Literal["hello world"] | None)
        x = "hello world"
        assert_type(x, Literal["hello world"])
    assert_type(x, Literal["hello world"] | None)
    "#,
);

testcase!(
    bug = "A recursive redefinition in a loop produces a hard-to-follow error message + location",
    test_while_creates_recursive_type,
    r#"
from typing import assert_type, Any, Literal
def f(condition) -> None:
    x = 1
    # It's fine to error here, but ideally we would error at the assignment
    # rather than the `while`, and ideally we would note that the type is recursive
    # in a way we don't support.
    while condition():  # E: `Literal[1] | list[int]` is not assignable to `int`
        assert_type(x, Literal[1] | list[int])
        x = [x]
        assert_type(x, list[int])
    assert_type(x, Literal[1] | list[int])
    "#,
);

testcase!(
    test_while_noop,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = 1
    while condition():
        pass
    assert_type(x, Literal[1])
    "#,
);

testcase!(
    test_while_fancy_noop,
    r#"
from typing import assert_type, Any, Literal
def f(condition) -> None:
    x = 1
    while condition():
        x = x
    assert_type(x, Literal[1])
    "#,
);

testcase!(
    test_while_if,
    r#"
from typing import assert_type, Any, Literal
def f(condition1, condition2) -> None:
    x = None
    while condition1():
        if condition2():
            x = "hello"
    assert_type(x, Literal['hello'] | None)
    "#,
);

testcase!(
    test_while_two_vars,
    r#"
from typing import assert_type, Any, Literal
def f(cond1, cond2, cond3) -> None:
    x = 1
    y = ""
    while cond1():
        if cond2():
            x = y
        if cond3():
            y = x
    assert_type(x, Literal["", 1])
    assert_type(y, Literal["", 1])
    "#,
);

testcase!(
    test_while_else,
    r#"
from typing import assert_type, Literal
def f(condition) -> None:
    x = None
    while condition():
        x = 1
    else:
        x = ""
    assert_type(x, Literal[""])
    "#,
);

testcase!(
    test_while_break_else,
    r#"
from typing import assert_type, Any, Literal
def f(cond1, cond2) -> None:
    x = None
    while cond1():
        if cond2():
            x = "value"
            break
        else:
            x = "overwritten"
    else:
        assert_type(x, Literal["overwritten"] | None)
        x = "default"
    assert_type(x, Literal["default", "value"])
    "#,
);

testcase!(
    test_while_else_while,
    r#"
while False:
    x = 0
else:
    while False:
        x = 1
    "#,
);

testcase!(
    test_while_infinite_implicit_return,
    r#"
def f1(b) -> int:
    while True:
        if b():
            return 1

def f2(b) -> int:  # E: Function declared to return `int` but is missing an explicit `return`
    while True:
        if b():
            break
    "#,
);

testcase!(
    test_while_reassignment_with_annotation,
    r#"
from typing import assert_type, Literal
def f(cond):
    x: int = 0
    while cond():
        x: int = 1
    assert_type(x, int)
    "#,
);

testcase!(
    test_for_simple,
    r#"
from typing import assert_type
def f(x: list[int]) -> None:
    for i in x:
        assert_type(i, int)
    assert_type(i, int)
    "#,
);

testcase!(
    test_for_tuple,
    r#"
from typing import assert_type
def f(x: tuple[int, str]) -> None:
    for i in x:
        assert_type(i, int | str)
    "#,
);

testcase!(
    test_for_literal_string,
    r#"
from typing import assert_type, LiteralString
for i in "abcd":
    assert_type(i, LiteralString)
    "#,
);

testcase!(
    test_for_any,
    r#"
from typing import Any, assert_type
def f(x: Any):
    for i in x:
        assert_type(i, Any)
    "#,
);

testcase!(
    test_for_reassign,
    r#"
from typing import assert_type
def f(x: list[int]):
    y = None
    for i in x:
        y = i
    assert_type(y, int | None)
    "#,
);

testcase!(
    test_for_else_reassign,
    r#"
from typing import assert_type, Literal
def f(x: list[int]):
    y = None
    for i in x:
        y = i
    else:
        y = 'done'
    assert_type(y, Literal['done'])
    "#,
);

testcase!(
    test_for_multiple_targets,
    r#"
from typing import assert_type
def f(x: list[tuple[int, str]]) -> None:
    for (i, j) in x:
        assert_type(i, int)
        assert_type(j, str)
    "#,
);

testcase!(
    test_for_scope,
    r#"
from typing import assert_type
def f(x: list[int]) -> None:
    for i in x:
        pass
    assert_type(i, int)
    "#,
);

testcase!(
    test_for_target_annot_compatible,
    r#"
def f(x: list[int]) -> None:
    i: int = 0
    for i in x:
        pass
    "#,
);

testcase!(
    test_for_target_annot_incompatible,
    r#"
def f(x: list[int]) -> None:
    i: str = ""
    for i in x: # E: Cannot use variable `i` with type `str` to iterate through `list[int]`
        pass
    "#,
);

fn loop_export_env() -> TestEnv {
    TestEnv::one(
        "imported",
        r#"
exported = None

for _ in []:
    ignored = 1
"#,
    )
}

testcase!(
    test_loop_export,
    loop_export_env(),
    r#"
import imported
from typing import assert_type

assert_type(imported.exported, None)
"#,
);

testcase!(
    test_loop_increment,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while cond:
        n += 1
    assert_type(n, int)
"#,
);

testcase!(
    test_loop_test_and_increment,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while n < 10:
        n += 1
    assert_type(n, int)
"#,
);

testcase!(
    test_nested_loop_increment,
    r#"
from typing import assert_type, Literal
def f_toplevel(cond: bool):
    n = "n"
    if cond:
        n = 1
    else:
        n = 1.5
    while cond:
        n += 1
    assert_type(n, float | int)
while True:
    # Make sure we treat a function nested in a loop the same
    # way (i.e. that the loop in a parent scope doesn't affect
    # flow merging in function scope).
    def f_in_loop(cond: bool):
        n = "n"
        if cond:
            n = 1
        else:
            n = 1.5
        while cond:
            n += 1
        assert_type(n, float | int)
"#,
);

testcase!(
    test_loop_test_and_increment_return,
    r#"
from typing import assert_type, Literal

def f(cond: bool):
    n = 1
    while cond:
        n += 1
    return n

assert_type(f(True), int)
"#,
);

testcase!(
    test_nested_loops_simple,
    r#"
def f(cond1: bool, cond2: bool):
    n = 0
    while cond1:
        while cond2:
            n += 1
"#,
);

testcase!(
    test_nested_loops_return,
    r#"
from typing import assert_type, Literal

def f(cond1: bool, cond2: bool):
    n = 0
    while cond1:
        while cond2:
            n += 1
    return n

assert_type(f(True, True), int)
"#,
);

testcase!(
    test_augassign_in_loop_simple,
    r#"
def f(args, cond):
    n = 0
    for arg in args:
        if cond:
            n += 1
"#,
);

testcase!(
    test_augassign_in_loop_return,
    r#"
from typing import assert_type, Literal

def f(args, cond):
    n = 0
    for arg in args:
        if cond:
            n += 1
    return n

assert_type(f([1, 2, 3], True), int)
"#,
);

testcase!(
    test_loops_and_ifs_galore,
    r#"
from typing import assert_type, Literal

def f(cond1: bool, cond2: bool, cond3: bool, cond4: bool):
    i = 0
    while cond1:
        if cond2:
            if cond3:
                pass
            if cond4:
                i += 1
    return i

assert_type(f(True, True, True, True), int)
"#,
);

testcase!(
    test_loop_defaulting,
    r#"
# From https://github.com/facebook/pyrefly/issues/104
from typing import assert_type
class Foo:
    pass

def rebase(parent: Foo | int) -> Foo: ...

def test(b: bool, x: Foo) -> None:
    while b:
        x = rebase(x)
    assert_type(x, Foo)
"#,
);

testcase!(
    test_loop_enumerate,
    r#"
# From https://github.com/facebook/pyrefly/issues/267
def foo() -> list[int]:
    results: list[int] = [1, 2, 3]
    for i, x in enumerate(results):
        results[i] = x * 10
    return results
"#,
);

testcase!(
    test_loop_nested_binding,
    r#"
# This used to fail, thinking the type was Never
def f():
    class X:
        pass

    while True:
        z = "" if True else ""
        break
    else:
        exit(1)

    x: X
"#,
);

testcase!(
    test_reveal_type_in_loop,
    r#"
# This used to get confused by what reveal_type is
from typing import *
x = 1
while True:
    reveal_type(x) # E: revealed type: Literal[1]
    break
else:
    exit(1)
"#,
);
