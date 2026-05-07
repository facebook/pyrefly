# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Stack[T]:
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...


def first[T](items: list[T]) -> T:
    return items[0]
