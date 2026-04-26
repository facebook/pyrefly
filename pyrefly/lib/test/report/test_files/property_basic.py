# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for @property getter/setter/deleter reporting.
# Ported from typestats TestProperty.


class Getter:
    @property
    def x(self) -> int:
        return 0


class GetterUntyped:
    @property
    def x(self):
        return 0


class GetterSetter:
    @property
    def x(self) -> int:
        return 0

    @x.setter
    def x(self, value: int) -> None:
        pass


class GetterSetterDeleter:
    @property
    def x(self) -> int:
        return 0

    @x.setter
    def x(self, value: int) -> None:
        pass

    @x.deleter
    def x(self) -> None:
        pass


class MultipleProperties:
    @property
    def a(self) -> int:
        return 0

    @property
    def b(self) -> str:
        return ""
