# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Schema class fields (dataclass, NamedTuple) are IMPLICIT (0 typable),
# but methods defined on those classes still count toward coverage.

from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Point:
    x: float
    y: float

    def distance(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


class Vector(NamedTuple):
    x: float
    y: float

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5
