# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List as MyList, TypeVar

T = TypeVar("T")
x = 42
y: Callable[[int], int] = lambda n: n
z: str = "hello"


def some_func() -> None:
    pass


class SomeClass:
    my_field = 42
