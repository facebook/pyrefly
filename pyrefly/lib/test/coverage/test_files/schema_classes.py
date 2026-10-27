# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for schema class detection: dataclass fields, enum members,
# TypedDict/NamedTuple fields should be counted as implicit (0 typable).
# Ported from typestats TestImplicitAttrs.

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, TypedDict


@dataclass
class Point:
    x: int
    y: float


class Color(Enum):
    RED = 1
    BLUE = 2


class Config(TypedDict):
    name: str
    value: int


class Coord(NamedTuple):
    x: int
    y: int
