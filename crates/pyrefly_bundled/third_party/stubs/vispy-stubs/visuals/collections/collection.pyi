import numpy as np
from numpy.typing import ArrayLike

from ... import gloo
from ...util.event import EventEmitter
from ..shaders import ModularProgram
from .base_collection import BaseCollection
from .util import fetchcode

# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

class Collection(BaseCollection):
    _gtypes: dict = ...

    def __init__(
        self,
        dtype: ArrayLike,
        itype: np.dtype | None,
        mode,
        vertex: str | tuple[str, ...],
        fragment: str | tuple[str, ...],
        program=None,
        **kwargs: str,
    ): ...
    def view(self, transform, viewport=None): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def draw(self, mode=None): ...

class CollectionView:
    def __init__(self, collection, transform=None, viewport=None): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def draw(self): ...
