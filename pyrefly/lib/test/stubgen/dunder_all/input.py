# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeAlias

__all__ = ["public_func", "PublicClass", "PUBLIC_VAR", "Vector"]

Vector: TypeAlias = list[float]
UNLISTED_ALIAS: TypeAlias = dict[str, int]


def public_func(x: int) -> str:
    return str(x)


def _private_helper() -> None:
    pass


def unlisted_func() -> bool:
    return True


class PublicClass:
    value: int

    def method(self) -> None:
        pass


class _PrivateClass:
    pass


class UnlistedClass:
    pass


PUBLIC_VAR: int = 20260317
_PRIVATE_VAR: str = "secret"
UNLISTED_VAR: float = 3.141529
