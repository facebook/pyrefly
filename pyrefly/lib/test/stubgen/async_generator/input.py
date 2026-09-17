# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from collections.abc import AsyncGenerator, AsyncIterator


async def annotated_async_generator() -> typing.AsyncGenerator[int, int]:
    yield 1


async def annotated_async_iterator() -> AsyncIterator[int]:
    yield 1


async def async_generator_with_yield_from() -> AsyncGenerator[int, None]:
    async for x in annotated_async_iterator():
        yield x


async def real_coroutine() -> int:
    return 1


async def coroutine_with_nested_generator() -> int:
    async def inner() -> AsyncGenerator[int, None]:
        yield 1

    return 1


def sync_generator() -> typing.Generator[int, None, None]:
    yield 1


class C:
    async def method_async_generator(self) -> AsyncGenerator[int, int]:
        yield 1

    async def method_coroutine(self) -> int:
        return 1
