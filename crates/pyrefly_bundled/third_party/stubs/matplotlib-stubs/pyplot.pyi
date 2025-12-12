import os
from collections.abc import Callable, Sequence
from datetime import datetime, tzinfo
from typing import Any, ContextManager, IO, Literal, overload, Unpack

import numpy as np
import numpy.typing as npt
from matplotlib import rcParams as rcParams
from matplotlib._stubs_utils._kwargs import FigureKW, GridspecKW, SubplotKW
from matplotlib._stubs_utils._properties import (
    Line2DProperty,
    PathCollectionProperties,
    RectangleProperties,
    TextPropertiesBase,
    TextPropertiesWithPositions,
    TitleTextProperties,
)
from matplotlib._stubs_utils._typing import ArrayLike, Color, FileLike, PathLike, Scalar
from matplotlib.axis import Tick
from matplotlib.colorizer import Colorizer
from matplotlib.contour import QuadContourSet
from matplotlib.layout_engine import LayoutEngine
from matplotlib.typing import ColorType
from typing_extensions import Self

from .artist import Artist
from .axes import Axes as Axes
from .backend_bases import FigureManagerBase, MouseButton
from .collections import (
    BrokenBarHCollection,
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
    QuadMesh,
)
from .colors import Colormap, Normalize
from .container import BarContainer, ErrorbarContainer, StemContainer
from .figure import Figure, SubFigure
from .image import AxesImage, FigureImage
from .legend import Legend
from .lines import Line2D
from .markers import MarkerStyle
from .patches import FancyArrow, Polygon
from .quiver import Quiver
from .scale import ScaleBase
from .streamplot import StreamplotSet
from .table import Table
from .text import Annotation, Text
from .transforms import Bbox, Transform
from .tri.tricontour import TriContourSet
from .widgets import SubplotTool

def install_repl_displayhook() -> None: ...
def uninstall_repl_displayhook() -> None: ...
def draw_all(force: bool = False) -> None: ...
def set_loglevel(*args, **kwargs) -> None: ...
def findobj(o=..., match=..., include_self: bool = ...) -> list: ...
def switch_backend(newbackend: str) -> None: ...
def new_figure_manager(*args, **kwargs) -> FigureManagerBase: ...
def draw_if_interactive(*args, **kwargs): ...
def show(*, block: bool | None = None) -> None: ...
def isinteractive() -> bool: ...

class _IoffContext:
    def __init__(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

class _IonContext:
    def __init__(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def ioff() -> _IoffContext: ...
def ion() -> _IonContext: ...
def pause(interval) -> None: ...
def rc(group, **kwargs) -> None: ...
def rc_context(rc: dict = ..., fname: str | PathLike = ...) -> ContextManager: ...
def rcdefaults() -> None: ...
def getp(obj: Artist, *args, **kwargs): ...
def get(obj: Artist, *args, **kwargs): ...
def setp(obj: Artist | list, *args, **kwargs): ...
def xkcd(scale: float = ..., length: float = ..., randomness: float = ...) -> _xkcd: ...

class _xkcd:
    def __init__(self, scale: float, length: float, randomness: float) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...

def figure(
    num: int | str | Figure | SubFigure | None = None,
    figsize: ArrayLike | tuple[float, float, Literal["in", "cm", "px"]] | None = None,
    dpi: float | None = None,
    *,
    facecolor: ColorType | None = None,
    edgecolor: ColorType | None = None,
    frameon: bool = True,
    FigureClass: type[Figure] = ...,
    clear: bool = False,
    layout: Literal["constrained", "compressed", "tight", "none"] | LayoutEngine | None = None,
    **kwargs: Any,
) -> Figure: ...
def gcf() -> Figure: ...
def fignum_exists(num) -> bool: ...
def get_fignums() -> list[int]: ...
def get_figlabels(): ...
def get_current_fig_manager() -> FigureManagerBase: ...
def connect(s: str, func: Callable): ...
def disconnect(cid: int): ...
def close(fig: None | int | str | Figure | Literal["all"] = None) -> None: ...
def clf() -> None: ...
def draw() -> None: ...
def savefig(
    fname: str | os.PathLike[str] | IO[bytes],
    *,
    transparent: bool | None = None,
    dpi: float | Literal["figure"] = "figure",
    format: str | None = None,
    metadata: dict[str, str] | None = None,
    bbox_inches: str | Bbox | None = None,
    pad_inches: float | Literal["layout"] = 0.1,
    facecolor: Color | Literal["auto"] = "auto",
    edgecolor: Color | Literal["auto"] = "auto",
    backend: str | None = None,
    **kwargs: Any,
) -> None: ...
def figlegend(*args, **kwargs) -> Legend: ...
def axes(arg: None | tuple = ..., **kwargs) -> Axes: ...
def delaxes(ax: Axes = ...) -> None: ...
def sca(ax: Axes) -> None: ...
def cla() -> None: ...
@overload
def subplots(
    nrows: Literal[1] = ...,
    ncols: Literal[1] = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[True] = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: SubplotKW | None = ...,
    gridspec_kw: GridspecKW | None = ...,
    **fig_kw: Unpack[FigureKW],
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[False],
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: SubplotKW | None = ...,
    gridspec_kw: GridspecKW | None = ...,
    **fig_kw: Unpack[FigureKW],
) -> tuple[Figure, npt.NDArray[Any]]: ...
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: SubplotKW | None = ...,
    gridspec_kw: GridspecKW | None = ...,
    **fig_kw: Unpack[FigureKW],
) -> tuple[Figure, Any]: ...
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
    sharey: bool | Literal["none", "all", "row", "col"] = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: SubplotKW | None = None,
    gridspec_kw: GridspecKW | None = None,
    **fig_kw: Unpack[FigureKW],
) -> tuple[Figure, Any]: ...
def subplot_mosaic(
    mosaic: list[list] | str,
    *,
    sharex: bool = ...,
    sharey: bool = ...,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    empty_sentinel: object = ...,
    subplot_kw: SubplotKW | None = None,
    per_subplot_kw: dict[str | tuple[str, str], SubplotKW] | None = None,
    gridspec_kw: GridspecKW | None = None,
    **fig_kw: Unpack[FigureKW],
) -> tuple[Figure, dict[Text, Axes]]: ...
def subplot2grid(
    shape: Sequence[int],
    loc: Sequence[int],
    rowspan: int = ...,
    colspan: int = ...,
    fig: Figure = ...,
    **kwargs,
) -> Axes: ...
def twinx(ax: Axes = ...) -> Axes: ...
def twiny(ax: Axes = ...) -> Axes: ...
def subplot_tool(targetfig: Figure = ...) -> SubplotTool: ...
def box(on: bool | None = ...): ...
@overload
def xlim() -> tuple[float, float]: ...
@overload
def xlim(
    left: None = None,
    right: None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    xmin: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    xmax: float | datetime | None = None,
) -> tuple[float, float]: ...
@overload
def xlim(
    left: None = None,
    right: float | datetime | None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    xmin: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    xmax: None = None,
) -> tuple[float, float]: ...
@overload
def xlim(
    left: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    right: None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    xmin: None = None,
    xmax: float | datetime | None = None,
) -> tuple[float, float]: ...
@overload
def xlim(
    left: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    right: float | datetime | None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    xmin: None = None,
    xmax: None = None,
) -> tuple[float, float]: ...
def xlim(*args, **kwargs) -> tuple[float, float]: ...
@overload
def ylim() -> tuple[float, float]: ...
@overload
def ylim(
    bottom: None = None,
    top: None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    ymin: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    ymax: float | datetime | None = None,
) -> tuple[float, float]: ...
@overload
def ylim(
    bottom: None = None,
    top: float | datetime | None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    ymin: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    ymax: None = None,
) -> tuple[float, float]: ...
@overload
def ylim(
    bottom: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    top: None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    ymin: None = None,
    ymax: float | datetime | None = None,
) -> tuple[float, float]: ...
@overload
def ylim(
    bottom: float | datetime | tuple[float, float] | tuple[datetime, datetime] | None = None,
    top: float | datetime | None = None,
    *,
    emit: bool = True,
    auto: bool | None = False,
    ymin: None = None,
    ymax: None = None,
) -> tuple[float, float]: ...
def ylim(*args, **kwargs) -> tuple[float, float]: ...
def xticks(ticks: ArrayLike | None = None, labels: Sequence[str] | None = None, *, minor: bool = False, **kwargs: Unpack[TextPropertiesBase]) -> tuple[list[Tick], list[Text]]: ...
def yticks(ticks: ArrayLike | None = None, labels: Sequence[str] | None = None, *, minor: bool = False, **kwargs: Unpack[TextPropertiesBase]) -> tuple[list[Tick], list[Text]]: ...
def rgrids(
    radii: Sequence[float] = ...,
    labels: Sequence[str] | None = ...,
    angle: float = ...,
    fmt: str | None = ...,
    **kwargs,
) -> tuple[list[Line2D], list[Text]]: ...
def thetagrids(
    angles: Sequence[float] = ...,
    labels: Sequence[str] | None = ...,
    fmt: str | None = ...,
    **kwargs,
) -> tuple[list[Line2D], list[Text]]: ...
def get_plot_commands() -> list: ...
def colorbar(mappable=..., cax: Axes = ..., ax: Axes = ..., **kwargs): ...
def clim(vmin: float | None = ..., vmax: float | None = ...): ...
def set_cmap(cmap: Colormap | str): ...
def imread(fname: str | FileLike, format: str = ...) -> np.ndarray: ...
def imsave(fname: str | PathLike | FileLike, arr: ArrayLike, **kwargs): ...
def matshow(
    A: ArrayLike,
    fignum: None | int | Literal[False] = ...,
    **kwargs,
) -> AxesImage: ...
def polar(*args, **kwargs): ...
def figimage(
    X: ArrayLike,
    xo: float = ...,
    yo: float = ...,
    alpha: None | float = ...,
    norm: Normalize = ...,
    cmap: str | Colormap = ...,
    vmin: float = ...,
    vmax: float = ...,
    origin: Literal["upper", "lower"] = ...,
    resize: bool = ...,
    **kwargs,
) -> FigureImage: ...
def figtext(
    x: float,
    y: float,
    s: str,
    fontdict: dict = ...,
    **kwargs,
) -> text.Text: ...
def gca(): ...
def gci(): ...
def ginput(
    n: int = ...,
    timeout: float = ...,
    show_clicks: bool = ...,
    mouse_add: MouseButton | None = ...,
    mouse_pop: MouseButton | None = ...,
    mouse_stop: MouseButton | None = ...,
) -> list[tuple]: ...
def subplots_adjust(
    left: float = ...,
    bottom: float = ...,
    right: float = ...,
    top: float = ...,
    wspace: float = ...,
    hspace: float = ...,
) -> None: ...
def suptitle(t: str, **kwargs) -> Text: ...
def tight_layout(
    *,
    pad: float = 1.08,
    h_pad: float | None = None,
    w_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
) -> None: ...
def waitforbuttonpress(timeout=...): ...
def acorr(x: ArrayLike, *, data=..., **kwargs): ...
def angle_spectrum(
    x: Sequence[float],
    Fs: float = ...,
    Fc: float = ...,
    window: Callable | np.ndarray = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    *,
    data=...,
    **kwargs,
): ...
def annotate(
    text: str,
    xy: Sequence[float],
    xytext: Sequence[float] = ...,
    xycoords: str | Artist | Transform | Callable = ...,
    textcoords: str | Artist | Transform | Callable = ...,
    arrowprops: dict = ...,
    annotation_clip: bool | None = ...,
    **kwargs,
) -> Annotation: ...
def arrow(x: float, y: float, dx: float, dy: float, **kwargs) -> FancyArrow: ...
def autoscale(
    enable: bool | None = ...,
    axis: Literal["both", "x", "y"] = ...,
    tight: bool | None = ...,
): ...
def axhline(
    y: float = ...,
    xmin: float = ...,
    xmax: float = ...,
    **kwargs: Unpack[Line2DProperty],
) -> Line2D: ...
def axhspan(
    ymin: float,
    ymax: float,
    xmin: float = ...,
    xmax: float = ...,
    **kwargs,
) -> Polygon: ...
def axis(*args, emit: bool = ..., **kwargs): ...
def axline(
    xy1: Sequence[float],
    xy2: Sequence[float] = ...,
    *,
    slope: float = ...,
    **kwargs,
) -> Line2D: ...
def axvline(
    x: float = ...,
    ymin: float = ...,
    ymax: float = ...,
    **kwargs,
) -> Line2D: ...
def axvspan(
    xmin: float,
    xmax: float,
    ymin: float = ...,
    ymax: float = ...,
    **kwargs,
) -> Polygon: ...
def bar(
    x: float | ArrayLike,
    height: float | ArrayLike,
    width: float | ArrayLike = ...,
    bottom: float | ArrayLike = ...,
    *,
    align: Literal["center", "edge"] = ...,
    data: None | Sequence[Any] = None,
    **kwargs: Unpack[RectangleProperties],
) -> BarContainer: ...
def barbs(*args, data=..., **kwargs): ...
def barh(
    y: float | ArrayLike,
    width: float | ArrayLike,
    height: float | ArrayLike = ...,
    left: float | ArrayLike = ...,
    *,
    align: Literal["center", "edge"] = ...,
    **kwargs,
) -> BarContainer: ...
def bar_label(
    container: BarContainer,
    labels: ArrayLike = ...,
    *,
    fmt: str = ...,
    label_type: Literal["edge", "center"] = ...,
    padding: float = ...,
    **kwargs,
) -> list: ...
def boxplot(
    x: ArrayLike,
    notch: bool = ...,
    sym: str = ...,
    vert: bool = ...,
    whis: float = ...,
    positions: ArrayLike = ...,
    widths: float | ArrayLike = ...,
    patch_artist: bool = ...,
    bootstrap: int = ...,
    usermedians=...,
    conf_intervals: ArrayLike = ...,
    meanline: bool = ...,
    showmeans=...,
    showcaps=...,
    showbox=...,
    showfliers=...,
    boxprops=...,
    labels: Sequence = ...,
    flierprops=...,
    medianprops=...,
    meanprops=...,
    capprops=...,
    whiskerprops=...,
    manage_ticks: bool = ...,
    autorange: bool = ...,
    zorder: float = ...,
    capwidths=...,
    *,
    data=...,
) -> dict: ...
def broken_barh(xranges, yrange, *, data=..., **kwargs) -> BrokenBarHCollection: ...
def clabel(CS, levels: ArrayLike = ..., **kwargs): ...
def cohere(
    x,
    y,
    NFFT: int = ...,
    Fs: float = ...,
    Fc: float = ...,
    detrend: Literal["none", "mean", "linear"] | Callable = ...,
    window: Callable | np.ndarray = ...,
    noverlap: float = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale_by_freq: bool = ...,
    *,
    data=...,
    **kwargs,
): ...
def contour(*args, data=..., **kwargs) -> QuadContourSet: ...
def contourf(*args, data=..., **kwargs) -> QuadContourSet: ...
def csd(
    x: Sequence[float],
    y: Sequence[float],
    NFFT: int = ...,
    Fs: float = ...,
    Fc: float = ...,
    detrend: Literal["none", "mean", "linear"] | Callable = ...,
    window: Callable | np.ndarray = ...,
    noverlap: float = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale_by_freq: bool = ...,
    return_line: bool = ...,
    *,
    data=...,
    **kwargs,
): ...
def errorbar(
    x: float | ArrayLike,
    y: float | ArrayLike,
    yerr: float | ArrayLike = ...,
    xerr: float | ArrayLike = ...,
    fmt: str = ...,
    ecolor: Color = ...,
    elinewidth: float = ...,
    capsize: float = ...,
    barsabove: bool = ...,
    lolims: bool = ...,
    uplims: bool = ...,
    xlolims: bool = ...,
    xuplims: bool = ...,
    errorevery: int = ...,
    capthick: float = ...,
    *,
    data=...,
    **kwargs,
) -> ErrorbarContainer: ...
def eventplot(
    positions: ArrayLike | Sequence[ArrayLike],
    orientation: Literal["horizontal", "vertical"] = ...,
    lineoffsets: float | ArrayLike = ...,
    linelengths: float | ArrayLike = ...,
    linewidths: float | ArrayLike = ...,
    colors: Color | list[Color] = ...,
    linestyles: str | tuple | list = ...,
    *,
    data=...,
    **kwargs,
) -> list: ...
def fill(*args, data=..., **kwargs) -> list[Polygon]: ...
def fill_between(
    x,
    y1: Scalar,
    y2: Scalar = ...,
    where: ArrayLike = ...,
    interpolate: bool = ...,
    step: Literal["pre", "post", "mid"] = ...,
    *,
    data=...,
    **kwargs,
) -> PolyCollection: ...
def fill_betweenx(
    y,
    x1: Scalar,
    x2: Scalar = ...,
    where: ArrayLike = ...,
    step: Literal["pre", "post", "mid"] = ...,
    interpolate: bool = ...,
    *,
    data=...,
    **kwargs,
) -> PolyCollection: ...
def grid(
    visible: bool | None = ...,
    which: Literal["major", "minor", "both"] = ...,
    axis: Literal["both", "x", "y"] = ...,
    **kwargs,
): ...
def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    C: ArrayLike = ...,
    gridsize: int = ...,
    bins: Literal["log"] | int | Sequence = ...,
    xscale: Literal["linear", "log"] = ...,
    yscale: Literal["linear", "log"] = ...,
    extent=...,
    cmap=...,
    norm=...,
    vmin=...,
    vmax=...,
    alpha=...,
    linewidths=...,
    edgecolors=...,
    reduce_C_function=...,
    mincnt: int = ...,
    marginals: bool = ...,
    *,
    data=...,
    **kwargs,
) -> PolyCollection: ...
def hist(
    x,
    bins: int | Sequence | str = ...,
    range: tuple | None = ...,
    density: bool = ...,
    weights=...,
    cumulative: bool | Literal[-1] = ...,
    bottom=...,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = ...,
    align: Literal["left", "mid", "right"] = ...,
    orientation: Literal["vertical", "horizontal"] = ...,
    rwidth: float | None = ...,
    log: bool = ...,
    color: Color | None = ...,
    label: str | None = ...,
    stacked: bool = ...,
    *,
    data=...,
    **kwargs,
): ...
def stairs(
    values: ArrayLike,
    edges: ArrayLike = ...,
    *,
    orientation: Literal["vertical", "horizontal"] = ...,
    baseline: float | ArrayLike | None = ...,
    fill: bool = ...,
    data=...,
    **kwargs,
): ...
def hist2d(
    x,
    y,
    bins: None | int | ArrayLike = ...,
    range=...,
    density: bool = ...,
    weights=...,
    cmin: float = ...,
    cmax: float = ...,
    *,
    data=...,
    **kwargs,
): ...
def hlines(
    y: float | ArrayLike,
    xmin: float | ArrayLike,
    xmax: float | ArrayLike,
    colors: list[Color] = ...,
    linestyles: Literal["solid", "dashed", "dashdot", "dotted"] = ...,
    label: str = ...,
    *,
    data=...,
    **kwargs,
) -> LineCollection: ...
def imshow(
    X: ArrayLike,
    cmap: str | Colormap = ...,
    norm: Normalize = ...,
    aspect: Literal["equal", "auto"] | float = ...,
    interpolation: str = ...,
    alpha: float | ArrayLike = ...,
    vmin: float = ...,
    vmax: float = ...,
    origin: Literal["upper", "lower"] = ...,
    extent: Sequence[float] = ...,
    *,
    interpolation_stage: Literal["data", "rgba"] = ...,
    filternorm: bool = ...,
    filterrad: float = ...,
    resample: bool = ...,
    url: str = ...,
    data=...,
    **kwargs,
) -> AxesImage: ...
def legend(handles: list[Artist | tuple[Artist, ...]] | None = None, labels: list[str] | None = None) -> Legend: ...
def locator_params(
    axis: Literal["both", "x", "y"] = ...,
    tight: bool | None = ...,
    **kwargs,
): ...
def loglog(
    *args: float | npt.ArrayLike | str | Sequence[datetime],
    base: float = 10,
    subs: Sequence[float] | None = None,
    nonpositive: Literal["mask", "clip"] = "clip",
    fmt: str | None = None,
    scalex: bool = True,
    scaley: bool = True,
    **kwargs: Unpack[Line2DProperty],
) -> list[Line2D]: ...
def magnitude_spectrum(
    x: Sequence,
    Fs: float = ...,
    Fc: float = ...,
    window: Callable | np.ndarray = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale: Literal["default", "linear", "dB"] = ...,
    *,
    data=...,
    **kwargs,
): ...
def margins(*margins, x: float = ..., y: float = ..., tight: bool | None = ...): ...
def minorticks_off(): ...
def minorticks_on(): ...
def pcolor(
    *args,
    shading: Literal["flat", "nearest", "auto"] = ...,
    alpha: float = ...,
    norm: Normalize = ...,
    cmap: str | Colormap = ...,
    vmin: float = ...,
    vmax: float = ...,
    data=...,
    **kwargs,
) -> Collection: ...
def pcolormesh(
    *args,
    alpha: float = ...,
    norm: Normalize = ...,
    cmap: str | Colormap = ...,
    vmin: float = ...,
    vmax: float = ...,
    shading: Literal["flat", "nearest", "gouraud", "auto"] = ...,
    antialiased=...,
    data=...,
    **kwargs,
) -> QuadMesh: ...
def phase_spectrum(
    x: Sequence,
    Fs: float = ...,
    Fc: float = ...,
    window: Callable | np.ndarray = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    *,
    data=...,
    **kwargs,
): ...
def pie(
    x,
    explode: ArrayLike = ...,
    labels: Sequence[str] = ...,
    colors: ArrayLike = ...,
    autopct: None | str | Callable = ...,
    pctdistance: float = ...,
    shadow: bool = ...,
    labeldistance: float | None = ...,
    startangle: float = ...,
    radius: float = ...,
    counterclock: bool = ...,
    wedgeprops: dict = ...,
    textprops: dict = ...,
    center: tuple[float, float] = ...,
    frame: bool = ...,
    rotatelabels: bool = ...,
    *,
    normalize: bool = ...,
    data=...,
): ...
def plot(
    *args: float | npt.ArrayLike | str | Sequence[datetime],
    fmt: str | None = None,
    scalex: bool = True,
    scaley: bool = True,
    **kwargs: Unpack[Line2DProperty],
) -> list[Line2D]: ...
def plot_date(
    x: ArrayLike,
    y: ArrayLike,
    fmt: str = ...,
    tz: tzinfo = ...,
    xdate: bool = ...,
    ydate: bool = ...,
    *,
    data=...,
    **kwargs,
) -> list: ...
def psd(
    x: Sequence,
    NFFT: int = ...,
    Fs: float = ...,
    Fc: float = ...,
    detrend: Literal["none", "mean", "linear"] | Callable = ...,
    window: Callable | np.ndarray = ...,
    noverlap: float = ...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale_by_freq: bool = ...,
    return_line: bool = ...,
    *,
    data=...,
    **kwargs,
): ...
def quiver(*args, data=..., **kwargs) -> Quiver: ...
def quiverkey(Q: Quiver, X: float, Y: float, U: float, label: str, **kwargs): ...
def scatter(
    x: float | ArrayLike,
    y: float | ArrayLike,
    s: float | ArrayLike = ...,
    c: ArrayLike | list[Color] | Color = ...,
    marker: MarkerStyle = ...,
    vmin: float = ...,
    vmax: float = ...,
    colorizer: Colorizer | None = None,
    plotnonfinite: bool = ...,
    data: dict[str, Any] = ...,
    **kwargs: Unpack[PathCollectionProperties],
) -> PathCollection: ...
def semilogx(*args, **kwargs) -> list: ...
def semilogy(*args, **kwargs) -> list: ...
def specgram(
    x: Sequence,
    NFFT: int = ...,
    Fs: float = ...,
    Fc: float = ...,
    detrend: Literal["none", "mean", "linear"] | Callable = ...,
    window: Callable | np.ndarray = ...,
    noverlap: int = ...,
    cmap: Colormap = ...,
    xextent=...,
    pad_to: float = ...,
    sides: Literal["default", "onesided", "twosided"] = ...,
    scale_by_freq: bool = ...,
    mode: Literal["default", "psd", "magnitude", "angle", "phase"] = ...,
    scale: Literal["default", "linear", "dB"] = ...,
    vmin=...,
    vmax=...,
    *,
    data=...,
    **kwargs,
): ...
def spy(
    Z,
    precision: float | Literal["present"] = ...,
    marker=...,
    markersize=...,
    aspect: Literal["equal", "auto"] | None | float = ...,
    origin: Literal["upper", "lower"] = ...,
    **kwargs,
) -> tuple[AxesImage, Line2D]: ...
def stackplot(
    x,
    *args,
    labels: list[str] = ...,
    colors: list[Color] = ...,
    baseline: Literal["zero", "sym", "wiggle", "weighted_wiggle"] = ...,
    data=...,
    **kwargs,
) -> list: ...
def stem(
    *args,
    linefmt: str = ...,
    markerfmt: str = ...,
    basefmt: str = ...,
    bottom: float = ...,
    label: str = ...,
    use_line_collection: bool = ...,
    orientation: str = ...,
    data=...,
) -> StemContainer: ...
def step(
    x: ArrayLike,
    y: ArrayLike,
    *args,
    where: Literal["pre", "post", "mid"] = ...,
    data=...,
    **kwargs,
) -> list: ...
def streamplot(
    x,
    y,
    u,
    v,
    density: float = ...,
    linewidth: float = ...,
    color: Color = ...,
    cmap: Colormap = ...,
    norm: Normalize = ...,
    arrowsize: float = ...,
    arrowstyle: str = ...,
    minlength: float = ...,
    transform=...,
    zorder: int = ...,
    start_points=...,
    maxlength: float = ...,
    integration_direction: Literal["forward", "backward", "both"] = ...,
    broken_streamlines: bool = ...,
    *,
    data=...,
) -> StreamplotSet: ...
def table(
    cellText=...,
    cellColours=...,
    cellLoc: Literal["left", "center", "right"] = ...,
    colWidths: list[float] = ...,
    rowLabels: list[str] = ...,
    rowColours: list[Color] = ...,
    rowLoc: Literal["left", "center", "right"] = ...,
    colLabels: list[str] = ...,
    colColours: list[Color] = ...,
    colLoc: Literal["left", "center", "right"] = ...,
    loc: str = ...,
    bbox: Bbox = ...,
    edges: Literal["open", "closed", "horizontal", "vertical"] = ...,
    **kwargs,
) -> Table: ...
def text(x: float, y: float, s: str, fontdict: TextPropertiesBase | None = None, **kwargs: Unpack[TextPropertiesBase]) -> Text: ...
def tick_params(axis: Literal["x", "y", "both"] = ..., **kwargs): ...
def ticklabel_format(
    *,
    axis: Literal["x", "y", "both"] = ...,
    style: Literal["sci", "scientific", "plain"] = ...,
    scilimits=...,
    useOffset: bool | float = ...,
    useLocale: bool = ...,
    useMathText: bool = ...,
): ...
def tricontour(*args, **kwargs) -> TriContourSet: ...
def tricontourf(*args, **kwargs) -> TriContourSet: ...
def tripcolor(
    *args,
    alpha=...,
    norm=...,
    cmap=...,
    vmin: float = ...,
    vmax: float = ...,
    shading: Literal["flat", "gouraud"] = ...,
    facecolors: ArrayLike = ...,
    **kwargs,
): ...
def triplot(*args, **kwargs): ...
def violinplot(
    dataset: ArrayLike,
    positions: ArrayLike = ...,
    vert: bool = ...,
    widths: ArrayLike = ...,
    showmeans: bool = ...,
    showextrema: bool = ...,
    showmedians: bool = ...,
    quantiles: ArrayLike = ...,
    points: int = ...,
    bw_method: str | Scalar | Callable = ...,
    *,
    data=...,
) -> dict: ...
def vlines(
    x: float | ArrayLike,
    ymin: float | ArrayLike,
    ymax: float | ArrayLike,
    colors: list[Color] = ...,
    linestyles: Literal["solid", "dashed", "dashdot", "dotted"] = ...,
    label: str = ...,
    *,
    data=...,
    **kwargs,
) -> LineCollection: ...
def xcorr(
    x,
    y,
    normed: bool = ...,
    detrend: Callable = ...,
    usevlines: bool = ...,
    maxlags: int = ...,
    *,
    data=...,
    **kwargs,
): ...
def sci(im): ...
def title(
    label: str,
    fontdict: dict[str, Any] | None = None,
    loc: Literal["left", "center", "right"] | None = None,
    pad: float | None = None,
    *,
    y: float | None = None,
    **kwargs: Unpack[TitleTextProperties],
) -> Text: ...
def xlabel(
    xlabel: str,
    fontdict: dict[str, Any] | None = None,
    labelpad: float | None = None,
    *,
    loc: Literal["left", "center", "right"] | None = None,
    **kwargs: Unpack[TextPropertiesWithPositions],
) -> Text: ...
def ylabel(
    ylabel: str,
    fontdict: dict[str, Any] | None = None,
    labelpad: float | None = None,
    *,
    loc: Literal["bottom", "center", "top"] | None = None,
    **kwargs: Unpack[TextPropertiesWithPositions],
) -> Text: ...
def xscale(
    value: Literal["linear", "log", "symlog", "logit"] | ScaleBase,
    **kwargs,
): ...
def yscale(
    value: Literal["linear", "log", "symlog", "logit"] | ScaleBase,
    **kwargs,
): ...
def autumn(): ...
def bone(): ...
def cool(): ...
def copper(): ...
def flag(): ...
def gray(): ...
def hot(): ...
def hsv(): ...
def jet(): ...
def pink(): ...
def prism(): ...
def spring(): ...
def summer(): ...
def winter(): ...
def magma(): ...
def inferno(): ...
def plasma(): ...
def viridis(): ...
def nipy_spectral(): ...
