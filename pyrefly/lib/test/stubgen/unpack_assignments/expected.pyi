# @generated
from typing import ClassVar, Literal

import os
from enum import Enum

A: Literal[1]
B: Literal['module']
PUBLIC: Literal['public']
FIRST: str
REST: list[str]
LX: int
LY: int


class C:
    A: ClassVar[Literal[3]]
    B: ClassVar[Literal['class']]


class Color(Enum):
    RED = ...
    BLUE = ...
