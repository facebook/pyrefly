# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import abstractmethod
from typing import overload

from typing_extensions import override


def noop(unused_param: int) -> None:
    pass


def not_implemented_raise(unused_param: int) -> None:
    raise NotImplementedError()


def not_implemented_return(unused_param: int) -> int:
    return NotImplemented


class A:
    @abstractmethod
    def method1(self, x: int) -> None:
        _ = 1

    @overload
    def method2(self, x: int): ...
    @overload
    def method2(self, x: str): ...
    def method2(self, x: object):
        return x


class B(A):
    @override
    def method1(self, x: int) -> None:
        _ = 1
