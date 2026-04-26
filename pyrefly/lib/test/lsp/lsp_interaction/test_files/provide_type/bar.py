# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NewType, overload, TypeAlias


class Bar:
    foo = 3

    class Baz:
        def f(self) -> None:
            self


Bar

Bar.Baz.f
Bar.Baz().f


def f_generic1[T: int = bool](t: T) -> T:
    t


def f_generic2[T: int = bool](t: T) -> T:
    inner: Callable[[T], T]


f_generic_callable = [f_generic1, f_generic2][0]


class A:
    def f(self):
        self


int.bit_length
any


def f_constraints[T: (bool, str) = bool](t: T) -> T: ...


class S:
    def f(self):
        asdf = super()


N = NewType("N", int)
x = N


@overload
def overloaded_func(x: None) -> None: ...
@overload
def overloaded_func(x: int) -> None: ...
def overloaded_func(x): ...


overloaded_func


class A[T]:
    def f1[F1](self):
        class B[T2]:
            def f2[F2](self, x: F1, y: F2, a: T, b: T2): ...

            f2


tup = (1, +2)

TA: TypeAlias = int | str


def alias() -> TA: ...


class Pos:
    def __pos__(self):
        return False


pos = Pos()
+pos
