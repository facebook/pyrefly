# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Foo:
    x: int | None


def f(foo: Foo) -> None:
    if foo.x is not None:
        y = foo.x
