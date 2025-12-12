from typing import Literal, TypedDict

from matplotlib._stubs_utils._typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.typing import ColorType

class SubplotKW(TypedDict, total=False):
    projection: Literal["aitoff", "hammer", "lambert", "mollweide", "polar", "rectilinear"] | str | None
    polar: bool
    axes_class: type
    sharex: Axes
    sharey: Axes
    label: str

class GridspecKW(TypedDict, total=False):
    nrows: int
    ncols: int
    figure: Figure | None
    left: float | None
    right: float | None
    top: float | None
    bottom: float | None
    wspace: float | None
    hspace: float | None
    width_ratios: ArrayLike | None
    height_ratios: ArrayLike | None

class FigureKW(TypedDict, total=False):
    num: int | str | Figure | SubFigure | None
    figsize: ArrayLike | tuple[float, float, Literal["in", "cm", "px"]] | None
    dpi: float | None
    facecolor: ColorType | None
    edgecolor: ColorType | None
    frameon: bool
    FigureClass: type[Figure]
    clear: bool
    layout: Literal["constrainedcompressed", "tight", "none"] | LayoutEngine | None

class NoKW(TypedDict): ...
