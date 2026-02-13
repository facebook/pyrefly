# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from ._color_dict import get_color_dict as get_color_dict, get_color_names as get_color_names
from .color_array import Color as Color, ColorArray as ColorArray
from .colormap import (
    BaseColormap as BaseColormap,
    Colormap as Colormap,
    get_colormap as get_colormap,
    get_colormaps as get_colormaps,
)

__all__ = [
    "Color",
    "ColorArray",
    "Colormap",
    "BaseColormap",
    "get_colormap",
    "get_colormaps",
    "get_color_names",
    "get_color_dict",
]
