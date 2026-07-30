# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests that Protocol classes have n_typable=0 for the class itself,
# but their methods still count toward coverage.

from typing import Protocol


class Drawable(Protocol):
    def draw(self, x: int, y: int) -> None: ...

    def resize(self, factor: float) -> bool: ...
