/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_text_size::Ranged;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::test::util::TestEnv;
use crate::testcase;

#[test]
fn test_with_narrow_query_roles_and_entry_sources() {
    let source = r#"
from contextlib import nullcontext, suppress

def suppressing(x: str | bytes) -> None:
    with suppress(AssertionError):
        assert isinstance(x, str)
        print(x)

def non_suppressing(x: str | bytes) -> None:
    with nullcontext():
        assert isinstance(x, str)
        print(x)

def unchanged(x: str | bytes) -> None:
    with suppress(AssertionError):
        pass
"#;
    let (state, handles) = TestEnv::one("main", source).to_state();
    let handle = handles("main");
    state.transaction().ad_hoc_solve(&handle, "with_query_roles", |answers| {
        let bindings = answers.bindings();
        let queries = bindings.keys::<Key>().filter(|idx| matches!(bindings.idx_to_key(*idx), Key::SuppressionNarrow(_))).collect::<Vec<_>>();
        assert_eq!(queries.len(), 2, "unchanged flow adds no suppression narrow query");
        for idx in queries {
            let key = bindings.idx_to_key(idx);
            let Binding::SuppressedException(query) = bindings.get(idx) else { panic!("suppression-only query data") };
            assert!(query.body.is_none());
            assert_eq!(query.contexts.len(), 1);
            let context = query.contexts[0];
            assert!(matches!(bindings.idx_to_key(context), Key::ContextExpr(_)), "exit comes from the original manager, not its as value");
            assert!(bindings.idx_to_key(context).range().start() > key.range().start());
            let known_false = bindings.module().code_at(bindings.idx_to_key(context).range()).starts_with("nullcontext");
            assert_eq!(answers.get_idx(idx).ty().is_never(), known_false);
            let completion = bindings.keys::<Key>().find(|candidate| matches!(bindings.idx_to_key(*candidate), Key::SuppressedException(range) if *range == key.range())).expect("normal completion query");
            assert_ne!(completion, idx);
            assert!(!answers.get_idx(completion).ty().is_never(), "normal body completion differs from suppression-only eligibility");
        }
    }).expect("query-role solve");
}

#[test]
fn test_with_narrow_following_loop_views() {
    let source = r#"
from contextlib import suppress

def f(x: str | bytes, again: bool) -> None:
    with suppress(AssertionError):
        assert isinstance(x, str)
    while again:
        if isinstance(x, bytes):
            pass
        again = False
"#;
    for _ in 0..2 {
        let (state, handles) = TestEnv::one("main", source).to_state();
        let handle = handles("main");
        state
            .transaction()
            .ad_hoc_solve(&handle, "with_loop_prior", |answers| {
                let bindings = answers.bindings();
                let mut inspected = false;
                for idx in bindings.keys::<Key>() {
                    if matches!(bindings.idx_to_key(idx), Key::Phi(name) if name.0.as_str() == "x")
                    {
                        let expected = answers.unions(vec![
                            answers.heap.mk_class_type(answers.stdlib.str().clone()),
                            answers.heap.mk_class_type(answers.stdlib.bytes().clone()),
                        ]);
                        assert_eq!(answers.get_idx(idx).ty(), &expected);
                        assert_eq!(
                            answers.get_idx(idx).ty(),
                            &expected,
                            "cached reread preserves the solved view"
                        );
                        inspected = true;
                    }
                }
                assert!(inspected, "source has a following-loop prior for x");
            })
            .expect("loop-prior solve");
    }
}

testcase!(
    bug = "Hard returns keep the BASE missing-return diagnostic",
    test_with_narrow_hard_return_baseline,
    r#"
from contextlib import suppress

def hard_return() -> int:  # E: paths are missing an explicit `return`
    with suppress(ValueError):
        return 1
    "#,
);

testcase!(
    test_with_narrow_plain,
    r#"
from contextlib import suppress
from typing import Literal, NoReturn

def may_raise() -> None: ...
def never() -> NoReturn:
    raise ValueError()
class TrueExit:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> Literal[True]:
        return True

def true_exit(x: str | bytes) -> str:
    with TrueExit():
        assert isinstance(x, str)
    return x  # E: is not assignable to declared return type

def asserted(x: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
    return x  # E: is not assignable to declared return type

def early(x: str | bytes) -> str:
    with suppress(ValueError):
        may_raise()
        assert isinstance(x, str)
    return x  # E: is not assignable to declared return type

def raised(x: str | bytes) -> str:
    with suppress(ValueError):
        if isinstance(x, str):
            pass
        else:
            raise ValueError()
    return x  # E: is not assignable to declared return type

def jump(x: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
        return x
    return x  # E: is not assignable to declared return type

def no_return(x: str | bytes) -> str:
    with suppress(ValueError):
        never()
        assert isinstance(x, str)  # E: This code is unreachable
    return x  # E: is not assignable to declared return type

def hard_break() -> int:
    for _ in [1]:
        with suppress(ValueError):
            break
    return 1

def hard_continue() -> int:
    for _ in [1]:
        with suppress(ValueError):
            continue
    return 1

def sibling(x: str | bytes, y: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
    return y  # E: is not assignable to declared return type

def prior(x: str | bytes) -> str:
    assert isinstance(x, str)
    with suppress(ValueError):
        may_raise()
    return x
    "#,
);

testcase!(
    test_with_narrow_read_exclusions,
    r#"
from contextlib import suppress

class C:
    value: int

def factory() -> C: ...

def subject_read(x: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
        print(x.upper())
    return x  # E: is not assignable to declared return type

def unknown_read(x: str | bytes, path: str) -> str:
    with suppress(AssertionError, OSError):
        assert isinstance(x, str)
        print(open(path).read())
        print("".join([x]))
        print(factory().value)
        print([1][0])
    return x  # E: is not assignable to declared return type

def nested_read(x: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
        def read() -> None:
            print(factory().value)
    return x  # E: is not assignable to declared return type
    "#,
);

testcase!(
    bug = "Write targets remain excluded from the bounded narrowing repair",
    test_with_narrow_write_target_controls,
    r#"
from contextlib import nullcontext, suppress

class C:
    value: int | str

def factory() -> C: ...
def throw() -> None:
    raise ValueError()

def assigned(obj: C) -> str:
    assert isinstance(obj.value, str)
    with suppress(ValueError):
        obj.value = 1
        throw()
    return obj.value  # E: is not assignable to declared return type

def subscript(items: list[int | str]) -> str:
    assert isinstance(items[0], str)
    with suppress(ValueError):
        items[0] = 1
        throw()
    return items[0]  # E: is not assignable to declared return type

def discarded(obj: C, choose: bool) -> str:
    assert isinstance(obj.value, str)
    with suppress(ValueError):
        if choose:
            obj.value = 1
            throw()
            return ""
        else:
            obj.value = 2
    return obj.value  # E: is not assignable to declared return type

def unknown_write(x: str | bytes) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
        factory().value = 1
    # Unknown write roots conservatively exclude all root repair in this with.
    return x

def target_forms(obj: C, table: list[int | str]) -> None:
    with suppress(ValueError):
        obj.value: int | str = 1  # E: Cannot annotate non-self attribute
        obj.value += 1
        table[0] += 1  # E: `+=` is not supported between `str` and
        del obj.value
        del table[0]
        for obj.value in [1]:
            pass
        with nullcontext(1) as table[0]:
            pass
    "#,
);

testcase!(
    bug = "Phi-root and facet narrowing remain outside the bounded repair",
    test_with_narrow_skipped_baseline,
    r#"
from contextlib import suppress

class C:
    value: int | str

def branchy(x: int | str | bytes, c: bool) -> str | bytes:
    with suppress(AssertionError):
        if c:
            assert isinstance(x, str)
        else:
            assert isinstance(x, bytes)
    # With x=1, either failing assertion is suppressed, so this returns an int.
    return x
bad: str | bytes = branchy(1, True)

def facet(obj: C) -> str:
    with suppress(AssertionError):
        assert isinstance(obj.value, str)
    return obj.value

def implicit_capture() -> str:
    with suppress(AssertionError):
        assert isinstance(global_value, str)
    return global_value
global_value: str | bytes = b""

def outer(x: str | bytes) -> str:
    def inner() -> str:
        nonlocal x
        with suppress(AssertionError):
            assert isinstance(x, str)
        return x
    return inner()
    "#,
);

testcase!(
    bug = "Intermediate with writes and body initialization keep BASE behavior",
    test_with_narrow_written_state_baseline,
    r#"
from contextlib import suppress

def throw() -> None:
    raise ValueError()

class EnterFail:
    def __init__(self, value: str) -> None: ...
    def __enter__(self) -> None:
        raise ValueError()
    def __exit__(self, *args: object) -> None: ...

class C:
    value: int | str

def middle() -> int:
    v = 0
    with suppress(ValueError):
        v = "middle"
        throw()
        v = 0
    return v

def try_middle() -> int:
    v = 0
    try:
        v = "middle"
        throw()
        v = 0
    except ValueError:
        pass
    return v

def prefix_return() -> int:
    v = 0
    with suppress(ValueError):
        v = "middle"
        throw()
        return 0
    return v  # E: is not assignable to declared return type

def header_write() -> int:
    v = 0
    with suppress(ValueError), EnterFail(v := "header"):
        v = 0
    return v

def missing() -> int:
    with suppress(ValueError):
        throw()
        w = 1
    return w

def rebound(x: int | str | bytes) -> str:
    assert isinstance(x, str)
    with suppress(ValueError):
        x = 1
        throw()
    return x  # E: is not assignable to declared return type

def deleted(x: str) -> str:
    with suppress(ValueError):
        del x
        throw()
    return x  # E: `x`

def invalidated(obj: C) -> str:
    assert isinstance(obj.value, str)
    with suppress(ValueError):
        obj.value = 1
        throw()
    return obj.value  # E: is not assignable to declared return type
    "#,
);

testcase!(
    test_with_narrow_exit_policy_controls,
    r#"
from contextlib import contextmanager
from collections.abc import Generator
from typing import Any, Literal

class FalseExit:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> Literal[False]:
        return False
class NoneExit:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...
class OptionalExit:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> bool | None:
        return True
class AnyExit:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> Any:
        return True
class GenericExit[T: (bool, None)]:
    def __init__(self, value: T) -> None:
        self.value: T = value
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> T:
        return self.value
@contextmanager
def manager() -> Generator[None]:
    yield None

def false_exit(x: str | bytes) -> str:
    with FalseExit():
        assert isinstance(x, str)
    return x

def none_exit(x: str | bytes) -> str:
    with NoneExit():
        assert isinstance(x, str)
    return x

# These exits preserve the ordinary BASE policy, not a non-suppression proof.
def optional_exit(x: str | bytes) -> str:
    with OptionalExit():
        assert isinstance(x, str)
    return x

def any_exit(x: str | bytes) -> str:
    with AnyExit():
        assert isinstance(x, str)
    return x

def generated_exit(x: str | bytes) -> str:
    with manager():
        assert isinstance(x, str)
    return x

def generic_exit[T: (bool, None)](x: T, y: T) -> None:
    with GenericExit(y):
        assert not isinstance(x, bool)
    return x
    "#,
);

testcase!(
    bug = "Loop-carried root narrows are conservatively skipped",
    test_with_narrow_manager_and_loop,
    r#"
from contextlib import nullcontext, suppress
from unittest import TestCase

class AsyncExit:
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *args: object) -> bool:
        return True

class T(TestCase):
    def test_as_target(self) -> None:
        with self.assertRaises(ValueError) as cm:
            raise ValueError()
        expected: BaseException = cm.exception

def first_as() -> int:
    with nullcontext(3) as value:
        pass
    return value

def combined(x: str | bytes, path: str) -> str:
    with suppress(OSError), open(path) as f:
        assert isinstance(x, str)
    f.close()  # Later-manager initialization keeps BASE behavior.
    return x  # E: is not assignable to declared return type

def nested(x: str | bytes) -> str:
    with suppress(AssertionError):
        with suppress(AssertionError):
            assert isinstance(x, str)
    return x  # E: is not assignable to declared return type

async def async_root(x: str | bytes) -> str:
    async with AsyncExit():
        assert isinstance(x, str)
    return x  # E: is not assignable to declared return type

def in_loop(x: str | bytes, again: bool) -> None:
    while again:
        with suppress(AssertionError):
            assert isinstance(x, str)
        text: str = x
        again = False

def following_loop(x: str | bytes, again: bool) -> str:
    with suppress(AssertionError):
        assert isinstance(x, str)
    while again:
        if isinstance(x, bytes):
            pass
        again = False
    return x  # E: is not assignable to declared return type
    "#,
);

testcase!(
    test_simple_with,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:
    assert_type(foo, int)

bar: str = "abc"
with Foo() as bar: # E: `int` is not assignable to variable `bar` with type `str`
    assert_type(bar, str)
    "#,
);

testcase!(
    test_simple_async_with,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    async def __aenter__(self) -> int:
        ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo:
        assert_type(foo, int)
    "#,
);

testcase!(
    test_simple_with_error,
    r#"
def test_sync() -> None:
    with 42 as foo:  # E: Cannot use `Literal[42]` as a context manager\n  Object of class `int` has no attribute `__enter__` # E: has no attribute `__exit__`
        pass

async def test_async() -> None:
    async with "abc" as bar:  # E: has no attribute `__aenter__` # E: has no attribute `__aexit__`
        pass
    "#,
);

testcase!(
    test_simple_with_wrong_enter_type,
    r#"
from types import TracebackType
class Foo:
    __enter__: int = 42
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:  # E: Expected `__enter__` to be a callable, got `int`
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_attribute_type,
    r#"
from types import TracebackType
class Foo:
    def __enter__(self) -> int: ...
    __exit__: int = 42

with Foo() as foo:  # E: Expected `__exit__` to be a callable, got `int`
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_argument_count,
    r#"
from typing import assert_type
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(self) -> None:
        ...

with Foo() as foo:  # E: Expected 0 positional arguments, got 3
    pass
    "#,
);

testcase!(
    test_with_wrong_exit_argument_type,
    r#"
from typing import assert_type
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(self, exc_type: int, exc_value: int, traceback: int) -> None:
        ...

with Foo() as foo: # E: `__exit__` must be callable with the argument types (type[BaseException], BaseException, TracebackType) # E: `__exit__` must be callable with the argument types (None, None, None)
    pass
    "#,
);

testcase!(
    test_with_wrong_return_type,
    r#"
from typing import assert_type
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> str:
        ...

with Foo() as foo:  # E: Cannot use `Foo` as a context manager\n  Return type `str` of function `Foo.__exit__` is not assignable to expected return type `bool | None`
    pass
    "#,
);

testcase!(
    test_async_with_dunder_aenter_not_async,
    r#"
from types import TracebackType
class Foo:
    def __aenter__(self) -> int:
        ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo: # E: Expected `__aenter__` to be async
        ...
"#,
);

testcase!(
    test_async_with_dunder_aexit_not_async,
    r#"
from types import TracebackType
class Foo:
    async def __aenter__(self) -> int:
        ...
    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

async def test() -> None:
    async with Foo() as foo: # E: Expected `__aexit__` to be async
        ...
"#,
);

testcase!(
    test_with_return_bool,
    r#"
class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> bool:
    return False

def f() -> int:  # E: missing an explicit `return`
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_true,
    r#"
from typing import Literal

class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> Literal[True]:
    return True

def f() -> int:  # E: missing an explicit `return`
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_false,
    r#"
# From https://github.com/facebook/pyrefly/issues/24

from typing import Literal

class CM:
  def __enter__(self) -> None:
    pass

  def __exit__(self, *args) -> Literal[False]:
    return False

def f() -> int:
  with CM():
    return 1
"#,
);

testcase!(
    test_with_return_any,
    r#"
# From https://github.com/facebook/pyrefly/issues/24

from typing import Any

def f(x: Any) -> int:
  with x:
    return 1
"#,
);

testcase!(
    test_with_contextmanager,
    r#"
import contextlib
from typing import Generator

@contextlib.contextmanager
def f() -> Generator[str, None, None]:
    yield ""

def g() -> bool:
    with f():
        return True
    "#,
);

testcase!(
    test_overloaded_exit_with,
    r#"
from typing import assert_type, overload
from types import TracebackType
class Foo:
    def __enter__(self) -> int:
        ...
    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
        /
    ) -> None:
        ...
    @overload
    def __exit__(
        self,
        exc_type: None,
        exc_value: None,
        traceback: None,
        /
    ) -> None:
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /
    ) -> None:
        ...

with Foo() as foo:
    assert_type(foo, int)
    "#,
);

testcase!(
    test_context_manager_exception_suppression_conformance,
    r#"
from typing import Any, Literal, assert_type

class CMBase:
    def __enter__(self) -> None:
        pass

class Suppress1(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return True

class Suppress2(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Literal[True]:
        return True

class NoSuppress1(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

class NoSuppress2(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        return False

class NoSuppress3(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> Any:
        return False

class NoSuppress4(CMBase):
    def __exit__(self, exc_type, exc_value, traceback) -> None | bool:
        return None

def suppress1(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress1():
            raise ValueError
    assert_type(x, int | str)

def suppress2(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress2():
            raise ValueError
    assert_type(x, int | str)

def no_suppress1(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress1():
            raise ValueError
    assert_type(x, str)

def no_suppress2(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress2():
            raise ValueError
    assert_type(x, str)

def no_suppress3(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress3():
            raise ValueError
    assert_type(x, str)

def no_suppress4(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress4():
            raise ValueError
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_multiple_items,
    r#"
from typing import assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def outer_suppresses(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress(), NoSuppress():
            raise ValueError
    assert_type(x, int | str)

def neither_suppresses(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress(), NoSuppress():
            raise ValueError
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_async,
    r#"
from typing import assert_type

class Suppress:
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, exc_type, exc_value, traceback) -> bool: ...

async def f(x: int | str) -> None:
    if isinstance(x, int):
        async with Suppress():
            raise ValueError
    assert_type(x, int | str)
"#,
);

testcase!(
    test_with_suppression_no_return_call,
    r#"
from typing import NoReturn, assert_type

def fail() -> NoReturn: ...

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def suppressed(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            fail()
    assert_type(x, int | str)

def not_suppressed(x: int | str) -> None:
    if isinstance(x, int):
        with NoSuppress():
            fail()
    assert_type(x, str)
"#,
);

testcase!(
    test_with_suppression_uninitialized,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def suppressed(c: bool) -> int:
    if c:
        y = 1
    else:
        with Suppress():
            raise ValueError
    return y  # E: `y` may be uninitialized

def not_suppressed(c: bool) -> int:
    if c:
        y = 1
    else:
        with NoSuppress():
            raise ValueError
    return y
"#,
);

// `__exit__` runs for `return`/`break`/`continue`, but its return value is only
// consulted when an exception is in flight, so a suppressing context manager cannot
// cancel them the way it cancels a `raise`.
testcase!(
    test_with_terminators_are_not_suppressible,
    r#"
from typing import assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def ret(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            return
    assert_type(x, str)

def brk(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                break
        assert_type(x, str)

def cont(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                continue
        assert_type(x, str)

def raises(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            raise ValueError
    assert_type(x, int | str)

def sys_exit(x: int | str) -> None:
    import sys
    if isinstance(x, int):
        with Suppress():
            sys.exit(1)
    assert_type(x, int | str)

def os_exit(x: int | str) -> None:
    import os
    if isinstance(x, int):
        with Suppress():
            os._exit(1)
    assert_type(x, str)

def ret_or_raise(c: bool, x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            if c:
                return
            else:
                raise ValueError
    assert_type(x, int | str)

def ret_or_ret(c: bool, x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            if c:
                return
            else:
                return
    assert_type(x, str)

# Entering the inner manager happens inside the outer's extent, so an exception from it can
# be suppressed and the `return` never reached, leaving this branch able to fall through.
def nested_ret(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                return
    assert_type(x, int | str)

def nested_raise(x: int | str) -> None:
    if isinstance(x, int):
        with Suppress():
            with Suppress():
                raise ValueError
    assert_type(x, int | str)
"#,
);

// An operation before a jump may raise. If the context manager suppresses that
// exception, the jump is never executed and control resumes after the `with`.
testcase!(
    test_with_exception_before_terminator_may_be_suppressed,
    r#"
from typing import TypeVar, assert_type

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Exploding:
    def __enter__(self) -> None:
        raise RuntimeError
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def may_raise() -> None: ...

def brk(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                may_raise()
                break
        assert_type(x, int | str)

def cont(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with Suppress():
                may_raise()
                continue
        assert_type(x, int | str)

def unreachable_in_with() -> None:
    with Suppress():
        return
        print("dead")  # E: This code is unreachable

def no_suppression(x: int | str) -> None:
    for _ in range(3):
        if isinstance(x, int):
            with NoSuppress():
                may_raise()
                break
        assert_type(x, str)

def fallback_after_suppressed_continue() -> None:
    while True:
        with Suppress():
            may_raise()
            continue
        print("the exception bypassed continue")
        break

def fallback_after_suppressed_break() -> None:
    while True:
        with Suppress():
            may_raise()
            break
        print("the exception bypassed break")

def no_operation_before_continue() -> None:
    while True:
        with Suppress():
            continue
        print("continue always executes")  # E: This code is unreachable

def raise_inside_a_special_export_assignment() -> None:
    # `TypeVar(...)` is bound by an early-returning arm of `stmt`; it may still raise.
    while True:
        with Suppress():
            T = TypeVar("T")
            break
        print("the exception bypassed the break")

def bare_return_always_executes() -> None:
    # A bare `return` evaluates nothing, so it cannot be bypassed, matching `break`.
    with Suppress():
        return
    print("return always executes")  # E: This code is unreachable

def no_operation_before_break() -> None:
    while True:
        with Suppress():
            break
        print("break always executes")  # E: This code is unreachable

def exploding_with() -> None:
    with Suppress(), Exploding():
        return
    print("reachable")

def exception_in_finally() -> None:
    with Suppress():
        try:
            return
        finally:
            may_raise()
    print("reachable")
"#,
);

// A statement that terminates the flow has still evaluated its header by then, so an
// enclosing context manager may suppress an exception from that header and skip the jump.
// Recording the header separately keeps this distinct from a bare `return`/`break`/`continue`,
// which evaluates nothing and stays unsuppressible per
// `test_with_terminators_are_not_suppressible`.
testcase!(
    test_with_exception_in_a_terminating_test_expression,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def may_raise_bool() -> bool: ...

def f() -> None:
    while True:
        with Suppress():
            if may_raise_bool():
                break
            else:
                break
        print("the exception bypassed both breaks")
"#,
);

// Entering a manager after the first can raise and be suppressed by an earlier one, so
// binding leaves the flow reachable after a `with` that enters more than one. The
// `__exit__` types settle whether any of them really suppresses, so the diagnostic is
// deferred to solving.
testcase!(
    test_dead_code_after_multi_manager_with,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def combined() -> None:
    with NoSuppress(), NoSuppress():
        return
    print("dead")  # E: This code is unreachable

def nested() -> None:
    with NoSuppress():
        with NoSuppress():
            return
    print("dead")  # E: This code is unreachable
"#,
);

// One suppressing manager anywhere in the chain is enough to keep the fall-through alive.
testcase!(
    test_live_code_after_multi_manager_with,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def outer_suppresses() -> None:
    with Suppress(), NoSuppress():
        return
    print("reachable")

def inner_suppresses() -> None:
    with NoSuppress(), Suppress():
        return
    print("reachable")
"#,
);

// Each terminating `with` in a suite is its own gate, and a statement runs only if every gate
// before it was passed. So a run of them is dead from the first `with` that cannot suppress,
// even when an earlier one could have.
testcase!(
    test_dead_code_after_a_run_of_withs,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def later_gate_is_closed() -> None:
    with Suppress(), NoSuppress():
        return
    with NoSuppress(), NoSuppress():
        return
    print("dead")  # E: This code is unreachable

def first_gate_is_closed() -> None:
    with NoSuppress(), NoSuppress():
        return
    with Suppress(), NoSuppress():  # E: This code is unreachable
        return
    print("dead")

def every_gate_is_open() -> None:
    with Suppress(), NoSuppress():
        return
    with NoSuppress(), Suppress():
        return
    print("reachable")
"#,
);

// A gated region that runs into definitely-dead code stops where the certain diagnostic takes
// over, so the two abut and neither swallows the other.
testcase!(
    test_gated_region_abuts_a_definitely_dead_one,
    r#"
class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def f() -> None:
    with NoSuppress(), NoSuppress():
        return
    print("a")  # E: This code is unreachable
    return
    print("b")  # E: This code is unreachable
"#,
);

// A `with` whose body exits under a static test must not make the following code dead: the
// exit only happens on other configurations. This is why the check is gated on the definite
// termination flag rather than on `has_terminated`, which a static test also sets.
testcase!(
    test_no_report_after_with_exited_by_static_test,
    r#"
import sys

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def asserted() -> None:
    with NoSuppress():
        assert sys.version_info >= (3, 20)
    print("runs on a new enough Python")

def gated_return() -> None:
    with NoSuppress(), NoSuppress():
        if sys.version_info < (3, 20):
            return
    print("runs on a new enough Python")
"#,
);

// Claiming code is dead requires knowing that no manager suppresses, which is stronger than
// failing to prove that one does. A manager we cannot read might suppress at runtime.
testcase!(
    test_no_report_after_with_when_suppression_is_unknown,
    r#"
import contextlib
from typing import Any

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def anything() -> Any: ...

def gradual() -> None:
    with anything():
        raise ValueError()
    print("an `Any` manager might suppress")

def bool_or_none() -> None:
    with contextlib.ExitStack(), NoSuppress():
        return
    print("`__exit__` returning `bool | None` might suppress")
"#,
);

// A `yield` is what makes a function a generator, so one in dead code is load-bearing and
// must not be blamed, exactly as in a definitely-dead region.
testcase!(
    test_no_report_of_generator_yield_after_with,
    r#"
from typing import Iterator

class NoSuppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

def never_yields() -> Iterator[int]:
    with NoSuppress(), NoSuppress():
        return
    yield 1

def reports_past_the_yields() -> Iterator[int]:
    with NoSuppress(), NoSuppress():
        return
    yield 1
    print("dead")  # E: This code is unreachable
"#,
);

// An `elif` test and a `case` guard are evaluated before their branch runs, exactly like the
// leading `if` test, so an exception from one is suppressible too. Spelling the same logic as
// `else: if ...` must not change the answer.
testcase!(
    test_with_exception_in_a_branch_header,
    r#"
class Suppress:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> bool: ...

def may_raise_bool() -> bool: ...

def elif_test(flag: bool) -> None:
    while True:
        with Suppress():
            if flag:
                break
            elif may_raise_bool():
                break
            else:
                break
        print("reachable")

def match_guard(n: int) -> None:
    while True:
        with Suppress():
            match n:
                case 1 if may_raise_bool():
                    break
                case _:
                    break
        print("reachable")

def match_pattern(n: object) -> None:
    while True:
        with Suppress():
            match n:
                case [1, 2]:
                    break
                case _:
                    break
        print("reachable")
"#,
);

// Recognizing that a manager suppresses is not confined to the reachability diagnostic: the
// same predicate decides whether a function can fall off the end of a `with`.
testcase!(
    test_with_overloaded_exit_affects_implicit_return,
    r#"
from types import TracebackType
from typing import overload

class Suppressing:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> None: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> bool: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

class NotSuppressing:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> bool: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> None: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

def falls_off_the_end() -> int:  # E: missing an explicit `return`
    with Suppressing():
        return 1

def cannot_fall_off_the_end() -> int:
    with NotSuppressing():
        return 1

# Only the overload taking exception arguments decides suppression, so the code after a `with`
# on the reversed manager really is dead.
def dead_after_reversed(x: int) -> None:
    with NotSuppressing():
        raise ValueError
    print("dead")  # E: This code is unreachable
"#,
);

// Overloads are the only way to spell "suppresses, but returns `None` on the normal path".
// Suppression is decided by the call made with exception arguments, so the overload selected
// there settles it; a plain `-> bool | None` remains non-suppressing (see `NoSuppress4`).
testcase!(
    test_with_suppression_overloaded_exit,
    r#"
from types import TracebackType
from typing import assert_type, overload

class CM:
    def __enter__(self) -> None: ...
    @overload
    def __exit__(self, t: None, v: None, tb: None) -> None: ...
    @overload
    def __exit__(self, t: type[BaseException], v: BaseException, tb: TracebackType) -> bool: ...
    def __exit__(self, t, v, tb) -> bool | None: ...

def f(x: int | str) -> None:
    if isinstance(x, int):
        with CM():
            raise ValueError
    assert_type(x, int | str)
"#,
);
