# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple


def f(t: tuple[int | None, str]) -> None:
    if t[0] is not None:
        y = t[0]
