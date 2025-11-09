# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import abstractmethod
from typing import overload, override


def noop(unused_param: int) -> None:
    pass


def not_implemented_raise(unused_param: int) -> None:
    raise NotImplementedError()


def not_implemented_return(unused_param: int) -> int:
    return NotImplemented


class A:
    @abstractmethod
    def method1(x: int) -> None:
        1

    @overload
    def method2(x: int): ...
    @overload
    def method2(x: str): ...
    def method2(x: object):
        return x


class B(A):
    @override
    def method1(x: int) -> None:
        1
