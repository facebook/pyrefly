# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
from .element import Element
from .transform import Transform

class Transformable(Element):
    def __init__(self, content=None, parent=None): ...
    @property
    def transform(self): ...
