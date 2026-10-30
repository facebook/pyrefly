# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Private names listed in __all__ should be reported as public (issue #3578).

__all__ = ["_foo", "_X", "_C"]


class _C:
    def method(self, x: int) -> int:
        return x


def _foo(x: int) -> int:
    return x


def _hidden(x: int) -> int:
    return x


_X: int = 1
_Y: int = 2
