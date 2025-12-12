# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from .base_filter import Filter as Filter
from .clipper import Clipper as Clipper
from .color import (
    Alpha as Alpha,
    ColorFilter as ColorFilter,
    IsolineFilter as IsolineFilter,
    ZColormapFilter as ZColormapFilter,
)
from .mesh import ShadingFilter as ShadingFilter, TextureFilter as TextureFilter, WireframeFilter as WireframeFilter
from .picking import PickingFilter as PickingFilter
