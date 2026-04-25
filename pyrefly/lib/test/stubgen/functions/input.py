# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def greet(name: str) -> str:
    return f"Hello, {name}!"


def add(x: int, y: int) -> int:
    return x + y


async def async_fetch(url: str) -> bytes:
    return b""


def with_defaults(x: int, y: int = 0, z: str = "hello") -> None:
    pass


def complex_defaults(x: int, data: object = object()) -> None:
    pass


def varargs(*args: int, **kwargs: str) -> None:
    pass


def pos_only(x: int, y: int, /) -> int:
    return x + y


def kw_only(*, key: str, value: int) -> None:
    pass
