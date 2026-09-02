# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests that inherited attrs are NOT double-counted.
# If class B(A) inherits field `x` from A, `x` should only appear
# under A's report, not under B as well.
#
# gh-3997: a reassigned attr inherits its base annotation's type quality.

from typing import Any


class Base:
    def __init__(self):
        self.x: int = 1
        self.y = "hello"
        self.a: Any = None


class Child(Base):
    def __init__(self):
        super().__init__()
        self.z: str = "child-only"


# Reassigns inherited fields unannotated: `x` typed (Base.x: int), `a` any (Base.a: Any).
class OverridingChild(Base):
    def __init__(self):
        self.x = 42
        self.a = object()
        self.w: float = 3.14


# gh-3997's exact shape: `attrs` is annotated only in the base class body (so it is
# not reported on the base) yet the subclass reassignment inherits its type, while
# `extra` is annotated nowhere and stays untyped.
class AttrBase:
    attrs: dict[str, Any]


class AttrChild(AttrBase):
    def __init__(self, attrs: dict[str, Any]) -> None:
        self.attrs = attrs
        self.extra = attrs
