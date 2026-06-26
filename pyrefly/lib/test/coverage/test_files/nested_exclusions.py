# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def outer() -> None:
    def inner() -> None:
        pass

    def inner2(x: int) -> str:
        return str(x)

    class LocalClass:
        def method(self) -> None:
            pass


def top_level(x: int) -> bool:
    return True


class TopLevel:
    def method(self) -> None:
        pass
