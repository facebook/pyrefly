# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Inner:
    value: int | None


class Outer:
    inner: Inner


def f(outer: Outer) -> None:
    if outer.inner.value is not None:
        y = outer.inner.value
