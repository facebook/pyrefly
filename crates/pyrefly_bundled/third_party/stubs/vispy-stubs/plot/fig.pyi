from ..scene import SceneCanvas
from ..util.svg.color import Color
from .plotwidget import PlotWidget

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class Fig(SceneCanvas):
    def __init__(self, bgcolor: Color | str = "w", size: tuple = ..., show: bool = True, keys="interactive", **kwargs): ...
    @property
    def plot_widgets(self): ...
    def __getitem__(self, idxs): ...
