import weakref
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .. import app, gloo
from ..app.application import Application
from ..app.canvas import Canvas, DrawEvent
from ..color import Color
from ..gloo.context import GLContext
from ..gloo.framebuffer import FrameBuffer
from ..scene.node import Node
from ..util import Frozen, logger
from ..util.event import Event
from ..util.profiler import Profiler
from ..visuals.transforms import TransformSystem
from ..visuals.visual import Visual
from .events import SceneMouseEvent
from .subscene import SubScene
from .visuals import VisualNode
from .widgets import Widget

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class SceneCanvas(app.Canvas, Frozen):
    def __init__(
        self,
        title: str = "VisPy canvas",
        size: tuple[float, float] = ...,
        position: tuple[float, float] | None = None,
        show: bool = False,
        autoswap: bool = True,
        app: str | Application | None = None,
        create_native: bool = True,
        vsync: bool = False,
        resizable: bool = True,
        decorate: bool = True,
        fullscreen: bool | int = False,
        config: Mapping | None = None,
        shared: None | Canvas | GLContext = None,
        keys: str | Mapping | None = None,
        parent: Widget | None = None,
        dpi: None | float = None,
        always_on_top: bool = False,
        px_scale: int = 1,
        bgcolor: Color | str = "black",
    ): ...
    @property
    def scene(self): ...
    @scene.setter
    def scene(self, node): ...
    @property
    def central_widget(self): ...
    @property
    def bgcolor(self): ...
    @bgcolor.setter
    def bgcolor(self, color): ...
    def update(self, node: Node | None = None): ...
    def on_draw(self, event: Event): ...
    def render(
        self,
        region: tuple | None = None,
        size: tuple | None = None,
        bgcolor: None | Color = None,
        crop: ArrayLike | None = None,
        alpha: bool = True,
    ) -> ArrayLike: ...
    def _draw_scene(self, bgcolor=None): ...
    def draw_visual(self, visual: Visual, event: DrawEvent | None = None): ...
    def _generate_draw_order(self, node=None): ...
    def _update_scenegraph(self, event): ...
    def _process_mouse_event(self, event): ...
    def visual_at(self, pos: tuple) -> None | Visual: ...
    def _visual_bounds_at(self, pos, node=None): ...
    def visuals_at(self, pos: tuple, radius: int = 10): ...
    def _render_picking(self, crop): ...
    def on_resize(self, event: Event): ...
    def on_close(self, event: Event): ...

    # -------------------------------------------------- transform handling ---
    def push_viewport(self, viewport: tuple): ...
    def pop_viewport(self): ...
    def push_fbo(self, fbo: FrameBuffer, offset: tuple, csize: tuple): ...
    def pop_fbo(self): ...
    def _current_framebuffer(self): ...
    def _update_transforms(self): ...
