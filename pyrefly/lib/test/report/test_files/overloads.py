# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for @overload decorated functions.
# Ported from typestats TestAnnotationCounts overload tests.

from typing import overload


@overload
def f(x: int) -> int: ...


@overload
def f(x: str) -> str: ...


def f(x):
    return x


class WithOverloads:
    @overload
    def method(self, x: int) -> int: ...

    @overload
    def method(self, x: str) -> str: ...

    def method(self, x):
        return x
