# @generated
import typing
from collections.abc import AsyncGenerator, AsyncIterator


def annotated_async_generator() -> typing.AsyncGenerator[int, int]: ...


def annotated_async_iterator() -> AsyncIterator[int]: ...


def async_generator_with_yield_from() -> AsyncGenerator[int, None]: ...


async def real_coroutine() -> int: ...


async def coroutine_with_nested_generator() -> int: ...


def sync_generator() -> typing.Generator[int, None, None]: ...


class C:
    def method_async_generator(self) -> AsyncGenerator[int, int]: ...

    async def method_coroutine(self) -> int: ...
