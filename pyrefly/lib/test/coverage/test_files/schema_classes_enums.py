# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Enum subclasses (IntEnum, StrEnum, Flag, IntFlag) members are IMPLICIT (0 typable).

from enum import auto, Flag, IntEnum, IntFlag, StrEnum


class Direction(IntEnum):
    NORTH = 1
    SOUTH = 2


class Color(StrEnum):
    RED = auto()
    GREEN = auto()


class Permission(Flag):
    READ = auto()
    WRITE = auto()


class Priority(IntFlag):
    LOW = 1
    HIGH = 4
