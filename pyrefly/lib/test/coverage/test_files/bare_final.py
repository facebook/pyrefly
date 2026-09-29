# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Bare `Final` should be typed: https://github.com/facebook/pyrefly/issues/3172

from typing import Final

golden: Final = 1.618033988749895
golden_ratio: Final = golden

pi: Final[float] = 3.14159

name: Final = "hello"


class Constants:
    def __init__(self):
        self.rate: Final = 0.05
        self.count: Final[int] = 10
