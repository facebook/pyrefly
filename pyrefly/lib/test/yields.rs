/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_generator,
    r#"
from typing import assert_type, Generator, Literal, Any

def yielding():
    yield 1

f = yielding()

next_f = next(f)
assert_type(next_f, int)
assert_type(f, Generator[Literal[1], Any, None])

"#,
);

testcase!(
    test_generator_with_return,
    r#"

from typing import assert_type, Generator, Literal, Any

def gen_with_return():
    yield 1
    yield 2
    return "done"

assert_type(gen_with_return(), Generator[Literal[1, 2], Any, Literal['done']])

"#,
);

testcase!(
    test_generator_send,
    r#"

from typing import Generator, assert_type

def accumulate(x: int) -> Generator[int, int, None]:
    yield x

gen = accumulate(10)
assert_type(gen, Generator[int, int, None])
gen.send(5)

"#,
);

testcase!(
    test_generator_send_inference,
    r#"

from typing import Generator, assert_type

class Yield: pass
class Send: pass
class Return: pass

def my_generator_nested() -> Generator[Yield, Send, Return]:
    yield Yield()
    return Return()

def my_generator() -> Generator[Yield, Send, Return]:
    s = yield Yield()
    y = yield from  my_generator_nested()

    assert_type(s, Send)
    assert_type(y, Return)

    return Return()

"#,
);

testcase!(
    test_nested_generator_annot,
    r#"
from typing import Generator, assert_type

class Y: pass
class S: pass
class R: pass

class Y2(Y): pass
class S2(S): pass

def foo(b: bool) -> Generator[Y2, S, R]:
    s1 = yield Y() # E: Type of yielded value `Y` is not assignable to declared return type `Y2`
    s2 = yield Y2()
    s3 = yield None # E: Type of yielded value `None` is not assignable to declared return type `Y2`

    assert_type(s1, S)
    assert_type(s2, S)
    assert_type(s3, S)

    if b:
        return None # E: Returned type `None` is not assignable to declared return type `R`
    else:
        return R()

def bar() -> Generator[Y, S2, None]:
    s1 = yield Y()
    s2 = yield Y2()
    s3 = yield None # E: Type of yielded value `None` is not assignable to declared return type `Y`

    r = yield from foo(True) # OK

    assert_type(s1, S2)
    assert_type(s2, S2)
    assert_type(s3, S2)
    assert_type(r, R)

def baz() -> Generator[Y2, S2, None]:
    s = yield from bar() # E: Cannot yield from a generator of type `Generator[Y, S2, None]` because it does not match the declared return type `Generator[Y2, S2, Unknown]`
    assert_type(s, None)

def qux() -> Generator[Y, S, None]:
    s = yield from bar() # E: Cannot yield from a generator of type `Generator[Y, S2, None]` because it does not match the declared return type `Generator[Y, S, Unknown]`
    assert_type(s, None)
"#,
);

testcase!(
    test_yield_with_iterator,
    r#"
from typing import Iterator, assert_type

def gen_numbers() -> Iterator[int]:
    yield 1
    yield 2
    yield 3

assert_type(gen_numbers(), Iterator[int])
"#,
);

testcase!(
    test_nested_generator_infer,
    r#"
from typing import Generator, assert_type, Literal, Any

def nested_generator():
    yield 1
    yield from another_generator()
    yield 3

def another_generator():
    yield 2

assert_type(nested_generator(), Generator[Literal[1, 2, 3], Any, None])
assert_type(another_generator(), Generator[Literal[2], Any, None])
"#,
);

testcase!(
    test_parametric_generator_type,
    r#"
from typing import Generator, TypeVar, assert_type

T = TypeVar('T')

def f(value: T) -> Generator[T, None, None]:
    while True:
        yield value

assert_type(f(3), Generator[int, None, None])
"#,
);

testcase!(
    test_async_generator_basic_type,
    r#"
from typing import AsyncGenerator, assert_type, Coroutine, Any

async def async_count_up_to() -> AsyncGenerator[int, None]:
    yield 2
assert_type(async_count_up_to(), AsyncGenerator[int, None])
"#,
);

// Normal async functions have their annotated return types "wrapped"
// into a coroutine. But async generators are annotated with their
// actual return type `AsyncGenerator`.
//
// At one point, the logic we were using to determine whether to wrap
// was based on checking the type, rather than a syntactic check. But
// this produces multiple issues, because the subtype check can pass
// when it shouldn't (e.g. if Any is in the return type, or if
// the return type is actually an `AsyncGenerator` *value* without the
// function itself being a generator).
//
// These tests are checks against regressing to that behavior.
testcase!(
    test_that_async_functions_are_not_incorrectly_treated_as_generators,
    r#"
from typing import AsyncGenerator, assert_type, Coroutine, Any, Never

async def async_any_or_none() -> Any | None:
    return 2
assert_type(async_any_or_none(), Coroutine[Any, Any, Any | None])

async def async_coroutine_of_async_generator() -> AsyncGenerator[int, None]:
    async def inner() -> AsyncGenerator[int, None]:
        yield 2
    return inner()
assert_type(async_coroutine_of_async_generator(), Coroutine[Any, Any, AsyncGenerator[int, None]])
"#,
);

testcase!(
    test_bare_yield,
    r#"
from typing import Generator

def bare_yield() -> Generator[int, None, None]:
    yield  # E: Expected to yield a value of type `int`

"#,
);

testcase!(
    test_async_infer_send,
    r#"
from typing import AsyncGenerator, assert_type

class Yield: pass
class Send: pass


async def my_generator() -> AsyncGenerator[Yield, Send]:
    s = yield Yield()

    assert_type(s, Send)

"#,
);

testcase!(
    test_async_error,
    r#"
from typing import AsyncGenerator, assert_type

class Yield: pass
class Send: pass


def my_generator() -> AsyncGenerator[Yield, Send]: # E: Generator function should return `Generator`
    s = yield Yield()
    assert_type(s, Send)

"#,
);

testcase!(
    test_async_generator_basic_inference,
    r#"
from typing import assert_type, Any, Literal, AsyncGenerator

async def async_count_up_to():
    yield 2
    return 4 # E: Return statement with value is not allowed in async generator
assert_type(async_count_up_to(), AsyncGenerator[Literal[2], Any])
"#,
);

testcase!(
    test_inferring_generators_that_return_generators,
    r#"
from typing import Any, Generator, assert_type

def generator() -> Generator[int, None, None]: ...

def generator2(x: int):
    yield x
    return generator()

assert_type(generator2(1), Generator[int, Any, Generator[int, None, None]])
"#,
);

testcase!(
    test_await_simple,
    r#"
from typing import Any, Awaitable, assert_type, Generator
class Foo(Awaitable[int]):
    def __await__(self) -> Generator[Any, Any, int]: ...
async def bar() -> str: ...

async def test() -> None:
    assert_type(await Foo(), int)
    assert_type(await bar(), str)
"#,
);

testcase!(
    test_await_literal,
    r#"
from typing import Any, Awaitable, Literal, Generator
class Foo(Awaitable[Literal[42]]):
    def __await__(self) -> Generator[Any, Any, Literal[42]]: ...
async def test() -> Literal[42]:
    return await Foo()
"#,
);

testcase!(
    test_await_non_awaitable,
    r#"
async def test() -> None:
    await 42  # E: Type `Literal[42]` is not awaitable
"#,
);

testcase!(
    test_await_wrong_await_return_type,
    r#"
class Foo:
    def __await__(self) -> int:
        ...

async def test() -> None:
    await Foo()  # E: Type `Foo` is not awaitable
"#,
);

testcase!(
    test_await_union_success,
    r#"
from typing import Awaitable, assert_type

async def test(x: Awaitable[int] | Awaitable[str]) -> None:
    y = await x
    assert_type(y, int | str)
"#,
);

testcase!(
    test_await_union_error,
    r#"
from typing import Any, Awaitable, assert_type

async def test(x: Awaitable[int] | str) -> None:
    y = await x  # E: Type `str` is not awaitable
    assert_type(y, int | Any)
"#,
);

testcase!(
    test_invalid_global_yield,
    r#"
yield 0  # E: Invalid `yield` outside of a function
yield from 0  # E: Invalid `yield from` outside of a function
"#,
);

testcase!(
    test_invalid_async_yield_from,
    r#"
from typing import AsyncGenerator
async def f() -> AsyncGenerator[int, None]:
    yield 1

async def g() -> AsyncGenerator[int, None]:
    yield from f() # E: Invalid `yield from` in async function
"#,
);

testcase!(
    test_missing_return,
    r#"
from typing import Generator
def f() -> Generator[None, None, int]:  # E: Function declared to return `int` but is missing an explicit `return`
    yield None
    "#,
);

testcase!(
    test_bad_return,
    r#"
from typing import Generator
def f() -> Generator[None, None, int]:
    yield None
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to declared return type `int`
    "#,
);

testcase!(
    test_async_iterate,
    r#"
from typing import AsyncGenerator, assert_type
async def gen() -> AsyncGenerator[int, None]:
    yield 2
async def test() -> None:
    async for x in gen():
        assert_type(x, int)
    async for y in [1, 2, 3]:  # E: Type `list[int]` is not an async iterable
        pass
    for z in gen():  # E: Type `AsyncGenerator[int, None]` is not iterable
        pass
"#,
);

testcase!(
    test_async_generator_comprehension_with_await,
    r#"
from typing import AsyncIterable, AsyncGenerator, assert_type

async def some_async_func(x: int) -> bool:
    return x % 2 == 0

async def main() -> None:
    generator = (x for x in [1, 2, 3] if await some_async_func(x))
    assert_type(generator, AsyncGenerator[int, None])
    async_iterable: AsyncIterable[int] = generator
    assert_type(generator, AsyncGenerator[int, None])
"#,
);

testcase!(
    test_implicit_async_generator,
    r#"
from typing import AsyncGenerator, Generator, assert_type

async def get_list() -> list[int]:
    return [1]

async def predicate() -> bool:
    return True

async def test_implicit_generators() -> None:
    assert_type((await predicate() for _ in [1]), AsyncGenerator[bool, None])
    assert_type((x for x in [1] if await predicate()), AsyncGenerator[int, None])
    assert_type((x for x in await get_list()), Generator[int, None, None])
    assert_type((x for _ in [1] for x in await get_list()), AsyncGenerator[int, None])
"#,
);

testcase!(
    test_async_generator_comprehension_with_nested_await,
    r#"
from typing import AsyncGenerator, assert_type

async def some_async_func(x: int) -> bool:
    return x % 2 == 0

async def main() -> None:
    # await is nested inside a comparison expression in the condition
    generator = (x for x in [1, 2, 3] if await some_async_func(x) == True)
    assert_type(generator, AsyncGenerator[int, None])
"#,
);

testcase!(
    bug = "We don't understand yield in lambda, and misattribute the yield to the surrounding function",
    test_lambda_yield,
    r#"
from typing import assert_type
def f(x: int):
    callback = lambda: (yield x)  # E: Invalid `yield` outside of a function
    l = [i for i in callback()]
    assert_type(l, list[int])  # E: assert_type(list[Any], list[int])
    return l
assert_type(f(1), list[int])  # E: assert_type(list[Any], list[int])
"#,
);

testcase!(
    test_generator_only_yield_from,
    r#"
from typing import Iterator
def generator_with_only_yield_from() -> Iterator[int]:
    yield from [1, 2, 3]
    "#,
);

testcase!(
    test_yield_iterator,
    r#"
from typing import Iterable, TypeVar, Iterator
_T = TypeVar("_T")

def f(start, iterable: Iterable[_T], step) ->  Iterator[_T]:
    next_i = start
    for i, element in enumerate(iterable):
        if i == next_i:
            yield element
            next_i += step

    "#,
);
