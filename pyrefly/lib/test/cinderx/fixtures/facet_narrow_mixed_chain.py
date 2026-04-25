# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Inner:
    value: int | None


def f(t: tuple[Inner, str]) -> None:
    if t[0].value is not None:
        y = t[0].value
