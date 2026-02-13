from typing import Literal

import numpy as np

from ...color import Color, Colormap
from ...util.event import Event
from ...visuals import ColorBarVisual
from .widget import Widget

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

class ColorBarWidget(Widget):
    def __init__(
        self,
        cmap: str | Colormap,
        orientation: Literal["left", "right", "top", "bottom"],
        label: str = "",
        label_color: str | Color = "black",
        clim: tuple[float, float] = ...,
        border_width: float = 0.0,
        border_color: str | Color = "black",
        padding: tuple = ...,
        axis_ratio: float = 0.05,
        **kwargs,
    ): ...
    def on_resize(self, event: Event): ...
    def _update_colorbar(self): ...
    def _calc_size(self): ...
    @property
    def cmap(self): ...
    @cmap.setter
    def cmap(self, cmap): ...
    @property
    def label(self): ...
    @label.setter
    def label(self, label): ...
    @property
    def ticks(self): ...
    @ticks.setter
    def ticks(self, ticks): ...
    @property
    def clim(self): ...
    @clim.setter
    def clim(self, clim): ...
    @property
    def border_color(self): ...
    @border_color.setter
    def border_color(self, border_color): ...
    @property
    def border_width(self): ...
    @border_width.setter
    def border_width(self, border_width): ...
    @property
    def orientation(self): ...
