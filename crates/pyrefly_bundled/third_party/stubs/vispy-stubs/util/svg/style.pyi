# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

from .color import Color
from .length import Length
from .number import Number

_converters: dict = ...

class Style:
    def __init__(self): ...
    def update(self, content): ...
    @property
    def xml(self): ...
    def _xml(self, prefix=""): ...
