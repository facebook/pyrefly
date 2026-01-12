# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------

import copy

from ...util import logger
from .base import namespace
from .path import Path
from .transformable import Transformable

class Group(Transformable):
    def __init__(self, content=None, parent=None): ...
    @property
    def flatten(self): ...
    @property
    def paths(self): ...
    def __repr__(self): ...
    @property
    def xml(self): ...
    def _xml(self, prefix=""): ...
