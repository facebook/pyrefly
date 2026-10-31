# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from typing import Callable, List as MyList, TypeVar  # noqa: F401

from schrodinger import cat

T = TypeVar("T")
x = 42
y: Callable[[int], int] = lambda n: n
z: str = "hello"
NO_RESULT = object()
EMPTY = frozenset()
DOUBLED = map(str, [1, 2])
NAMES = list[str]()
QUEUE = collections.deque[int]()
CATS = cat[None]()


def some_func() -> None:
    pass


class SomeClass:
    my_field = 42
