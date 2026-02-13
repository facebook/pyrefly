from collections.abc import Mapping

from numpy.typing import ArrayLike

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

def get_color_names() -> ArrayLike: ...
def get_color_dict() -> Mapping: ...

# This is used by color functions to translate user strings to colors
# For now, this is web colors, and all in hex. It will take some simple
# but annoying refactoring to deal with non-hex entries if we want them.

# Add the CSS colors, courtesy MIT-licensed code from Dave Eddy:
# github.com/bahamas10/css-color-names/blob/master/css-color-names.json

_color_dict: dict = ...
