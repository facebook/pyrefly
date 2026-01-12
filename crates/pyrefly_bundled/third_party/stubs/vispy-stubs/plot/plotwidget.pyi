from typing import Literal

from numpy.typing import ArrayLike, NDArray

from .. import scene
from ..color import Color, Colormap
from ..geometry import MeshData
from ..io import read_mesh
from ..scene.visuals import Image, LinePlot, Mesh, Polygon, Spectrogram, Volume
from ..scene.widgets.colorbar import ColorBarWidget

# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

__all__ = ["PlotWidget"]

class PlotWidget(scene.Widget):
    def __init__(self, *args, **kwargs): ...
    def _configure_2d(self, fg_color=None): ...
    def _configure_3d(self): ...
    def histogram(
        self,
        data: ArrayLike,
        bins: ArrayLike | int = 10,
        color: Color | str = "w",
        orientation: Literal["h", "v"] = "h",
    ) -> Polygon: ...
    def image(
        self, data: NDArray, cmap: str = "cubehelix", clim: str | tuple = "auto", fg_color: Color | None = None, **kwargs
    ) -> Image: ...
    def mesh(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        vertex_colors: ArrayLike | None = None,
        face_colors: ArrayLike | None = None,
        color: Color = ...,
        fname: str | None = None,
        meshdata: None | MeshData = None,
        shading: str = "auto",
    ) -> Mesh: ...
    def plot(
        self,
        data,
        color: Color | str = "k",
        symbol: str | None = None,
        line_kind: str = "-",
        width: float = 1.0,
        marker_size: float = 10.0,
        edge_color: Color | str = "k",
        face_color: Color | str = "b",
        edge_width: float = 1.0,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        connect: str | ArrayLike = "strip",
    ) -> LinePlot: ...
    def spectrogram(
        self,
        x: ArrayLike,
        n_fft: int = 256,
        step: None | int = None,
        fs: float = 1.0,
        window: str | None = "hann",
        normalize: bool = False,
        color_scale: Literal["linear", "log"] = "log",
        cmap: str = "cubehelix",
        clim: str | tuple = "auto",
    ) -> Spectrogram: ...
    def volume(
        self,
        vol: NDArray,
        clim: tuple[float, float] | None = None,
        method: Literal["mip", "iso", "translucent", "additive"] = "mip",
        threshold: float | None = None,
        cmap: str = "grays",
        **kwargs,
    ) -> Volume: ...
    def surface(self, zdata: ArrayLike, **kwargs): ...
    def colorbar(
        self,
        cmap: str | Colormap,
        position: Literal["left", "right", "top", "bottom"] = "right",
        label: str = "",
        clim: tuple[float, float] = ...,
        border_width: float = 0.0,
        border_color: str | Color = "black",
        **kwargs,
    ) -> ColorBarWidget: ...
