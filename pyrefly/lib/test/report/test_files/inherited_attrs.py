# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests that inherited attrs are NOT double-counted.
# If class B(A) inherits field `x` from A, `x` should only appear
# under A's report, not under B as well.


class Base:
    def __init__(self):
        self.x: int = 1
        self.y = "hello"


class Child(Base):
    def __init__(self):
        super().__init__()
        self.z: str = "child-only"


# Child re-assigns parent field without super().__init__()
class OverridingChild(Base):
    def __init__(self):
        self.x = 42
        self.w: float = 3.14
