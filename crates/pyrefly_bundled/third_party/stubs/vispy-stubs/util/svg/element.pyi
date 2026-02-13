# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
import copy

from .style import Style

namespace: str = ...

class Element:
    def __init__(self, content=None, parent=None): ...
    @property
    def root(self): ...
    @property
    def parent(self): ...
    @property
    def style(self): ...
    @property
    def viewport(self): ...
