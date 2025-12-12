# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import math
import re

from .. import logger
from .base import units

class Length:
    def __init__(self, content, mode="x", parent=None): ...
    def __float__(self): ...
    @property
    def value(self): ...
    def __repr__(self): ...

class XLength(Length):
    def __init__(self, content, parent=None): ...

class YLength(Length):
    def __init__(self, content, parent=None): ...

class XYLength(Length):
    def __init__(self, content, parent=None): ...
