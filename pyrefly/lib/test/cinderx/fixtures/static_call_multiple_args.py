# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __static__ import double, int64


def baz(x: int64, y: str, z: double) -> None:
    pass


baz(42, "hello", 3.14)
