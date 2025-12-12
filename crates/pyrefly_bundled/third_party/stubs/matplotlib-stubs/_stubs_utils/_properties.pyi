from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypedDict

import numpy.typing as npt
from matplotlib._enums import CapStyle, JoinStyle
from matplotlib._stubs_utils._typing import RGBAColor
from matplotlib.artist import Artist
from matplotlib.backend_bases import Event
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import BoxStyle
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.transforms import BboxBase, Transform
from matplotlib.typing import ColorType

class RectangleProperties(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: float
    angle: float
    animated: bool
    antialiased: bool | None
    aa: bool | None
    bounds: tuple[float, float, float, float]
    capstyle: CapStyle | Literal["butt", "projecting", "round"]
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Path | tuple[Path, Transform] | None
    color: ColorType
    edgecolor: ColorType | None
    ec: ColorType | None
    facecolor: ColorType | None
    fc: ColorType | None
    figure: Figure | SubFigure
    fill: bool
    gid: str
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    hatch_linewidth: float
    # height: float
    in_layout: bool
    joinstyle: JoinStyle | Literal["miter", "round", "bevel"]
    label: object
    linestyle: Literal["-", "--", "-.", ":", ""] | tuple[float, list[float]] | None
    ls: Literal["-", "--", "-.", ":", ""] | tuple[float, list[float]] | None
    linewidth: float | None
    lw: float | None
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: bool | float | None | Callable
    rasterized: bool
    sketch_params: SketchParams
    snap: bool | None
    transform: Transform
    url: str
    visible: bool
    # width: float
    # x: float
    xy: tuple[float, float]
    y: float
    zorder: float

class PatchProperties(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: float
    animated: bool
    antialiased: bool | None
    aa: bool | None
    capstyle: CapStyle | Literal["butt", "projecting", "round"]
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Path | tuple[Path, Transform] | None
    color: ColorType
    edgecolor: ColorType | None
    ec: ColorType | None
    facecolor: ColorType | None
    fc: ColorType | None
    figure: Figure | SubFigure
    fill: bool
    gid: str
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    hatch_linewidth: float
    in_layout: bool
    joinstyle: JoinStyle | Literal["miter", "round", "bevel"]
    label: object
    linestyle: Literal["-", "--", "-.", ":", ""] | tuple[float, list[float]] | None
    ls: Literal["-", "--", "-.", ":", ""] | tuple[float, list[float]] | None
    linewidth: float | None
    lw: float | None
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: bool | float | None | Callable
    rasterized: bool
    sketch_params: SketchParams
    snap: bool | None
    transform: Transform
    url: str
    visible: bool
    zorder: float

class FancyBboxPatchProperties(PatchProperties, total=False):
    xy: tuple[float, float]
    width: float
    height: float
    boxstyle: str | BoxStyle
    mutation_scale: float
    mutation_aspect: float

class SketchParams(TypedDict, total=False):
    scale: float
    length: float
    randomness: float

class Clim(TypedDict, total=False):
    vim: float
    vmax: float

class PathCollectionProperties(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: npt.ArrayLike | float | None
    animated: bool
    antialiased: bool | list[bool]
    aa: bool | list[bool]
    antialiaseds: bool | list[bool]
    array: npt.ArrayLike | None
    capstyle: CapStyle | Literal["butt", "projecting", "round"]
    clim: Clim
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Path | tuple[Path, Transform] | None
    cmap: Colormap | str | None
    color: ColorType | list[tuple[RGBAColor, ...]]
    edgecolor: ColorType | list[ColorType] | Literal["face"]
    ec: ColorType | list[ColorType] | Literal["face"]
    edgecolors: ColorType | list[ColorType] | Literal["face"]
    facecolor: ColorType | list[ColorType]
    facecolors: ColorType | list[ColorType]
    fc: ColorType | list[ColorType]
    figure: Figure | SubFigure
    gid: str
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    hatch_linewidth: Any
    in_layout: bool
    joinstyle: JoinStyle | Literal["miter", "round", "bevel"]
    label: object
    linestyle: str | tuple[str] | list[str]
    linestyles: str | tuple[str] | list[str]
    ls: str | tuple[str] | list[str]
    dashes: str | tuple[str] | list[str]
    linewidth: float | list[float]
    linewidths: float | list[float]
    lw: float | list[float]
    mouseover: bool
    norm: Normalize | str | None
    offset_transform: Transform
    transOffset: Transform
    offsets: npt.ArrayLike
    path_effects: list[AbstractPathEffect]
    paths: Any
    picker: bool | float | None | Callable
    pickradius: float
    rasterized: bool
    sizes: npt.NDArray | None
    sketch_params: SketchParams
    snap: bool | None
    transform: Transform
    url: str
    utls: list[str] | None
    visible: bool
    zorder: float

class Line2DProperty(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    aa: bool
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Path | tuple[Path, Transform] | None
    color: ColorType
    c: ColorType
    dash_capstyle: CapStyle | Literal["butt", "projecting", "round"]
    dash_joinstyle: JoinStyle | Literal["miter", "round", "bevel"]
    dashes: Sequence[float] | tuple[None, None]
    data: npt.NDArray | tuple[npt.NDArray]
    drawstyle: Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
    ds: Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
    figure: Figure | SubFigure
    fillstyle: Literal["full", "left", "right", "bottom", "top", "none"]
    gapcolor: ColorType | None
    gid: str
    in_layout: bool
    label: object
    linestyle: Literal["-", "solid", "--", "dashed", "-.", "dashdot", ":", "dotted", "", "none", "None", " "] | tuple[int, Sequence[int]]
    ls: Literal["-", "solid", "--", "dashed", "-.", "dashdot", ":", "dotted", "", "none", "None", " "] | tuple[int, Sequence[int]]
    linewidth: float
    lw: float
    marker: str | Path | MarkerStyle
    markeredgecolor: ColorType
    mec: ColorType
    markeredgewidth: float
    mew: float
    markerfacecolor: ColorType
    mfc: ColorType
    markerfacecoloralt: ColorType
    mfcalt: ColorType
    markersize: float
    ms: float
    markevery: None | int | tuple[int, int] | slice | list[int] | float | tuple[float, float] | list[bool]
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: float | Callable[[Artist, Event], tuple[bool, dict]]
    pickradius: float
    rasterized: bool
    sketch_params: SketchParams
    snap: bool | None
    solid_capstyle: CapStyle | Literal["butt", "projecting", "round"]
    solid_joinstyle: JoinStyle | Literal["miter", "round", "bevel"]
    transform: Any
    url: str
    visible: bool
    xdata: npt.NDArray
    ydata: npt.NDArray
    zorder: float

class TitleTextProperties(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    backgroundcolor: ColorType
    bbox: FancyBboxPatchProperties
    clip_box: Any
    clip_on: Any
    clip_path: Path
    color: ColorType
    c: ColorType
    figure: Figure | SubFigure
    fontfamily: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    family: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    fontfname: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    fontproperties: dict[str, Any] | str | Path
    font: dict[str, Any] | str | Path
    font_properties: dict[str, Any] | str | Path
    fontsize: float | Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
    size: float | Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
    fontstretch: (
        int
        | Literal[
            "ultra-condensed",
            "extra-condensed",
            "condensed",
            "semi-condensed",
            "normal",
            "semi-expanded",
            "expanded",
            "extra-expanded",
            "ultra-expanded",
        ]
    )
    stretch: (
        int
        | Literal[
            "ultra-condensed",
            "extra-condensed",
            "condensed",
            "semi-condensed",
            "normal",
            "semi-expanded",
            "expanded",
            "extra-expanded",
            "ultra-expanded",
        ]
    )
    fontstyle: Literal["normal", "italic", "oblique"]
    style: Literal["normal", "italic", "oblique"]
    fontvariant: Literal["normal", "small-caps"]
    variant: Literal["normal", "small-caps"]
    fontweight: (
        int
        | Literal[
            "ultralight",
            "light",
            "normal",
            "regular",
            "book",
            "medium",
            "roman",
            "semibold",
            "demibold",
            "demi",
            "bold",
            "heavy",
            "extra bold",
            "black",
        ]
    )
    weight: (
        int
        | Literal[
            "ultralight",
            "light",
            "normal",
            "regular",
            "book",
            "medium",
            "roman",
            "semibold",
            "demibold",
            "demi",
            "bold",
            "heavy",
            "extra bold",
            "black",
        ]
    )
    gid: str
    horizontalalignment: Literal["center", "right", "left"]
    ha: Literal["center", "right", "left"]
    in_layout: bool
    linespacing: float
    math_fontfamily: str
    mouseover: bool
    multialignment: Literal["left", "center", "right"]
    ma: Literal["left", "center", "right"]
    parse_math: bool
    path_effects: list[AbstractPathEffect]
    picker: bool | float | None | Callable
    position: tuple[float, float]
    rasterized: bool
    rotation: float | Literal["vertical", "horizontal"]
    rotation_mode: Literal["default", "anchor"] | None
    sketch_params: SketchParams
    snap: bool | None
    text: Any
    transform: Transform
    transform_rotates_text: bool
    url: str
    usetex: bool
    verticalalignment: Literal["center", "top", "bottom", "baseline", "center_baseline"]
    va: Literal["center", "top", "bottom", "baseline", "center_baseline"]
    visible: bool
    wrap: bool
    x: float
    zorder: float

class TextPropertiesBase(TypedDict, total=False):
    agg_filter: Callable[[npt.NDArray, float], tuple[npt.NDArray, int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    backgroundcolor: ColorType
    bbox: FancyBboxPatchProperties
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Path | tuple[Path, Transform] | None
    color: ColorType
    c: ColorType
    figure: Figure | SubFigure
    fontfamily: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    family: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    fontname: str | Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    fontproperties: dict[str, Any] | str | Path
    font: dict[str, Any] | str | Path
    font_properties: dict[str, Any] | str | Path
    fontsize: float | Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
    size: float | Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
    fontstretch: int | Literal["ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal", "semi-expanded", "expanded", "extra-expanded", "ultra-expanded"]
    stretch: int | Literal["ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal", "semi-expanded", "expanded", "extra-expanded", "ultra-expanded"]
    fontstyle: Literal["normal", "italic", "oblique"]
    style: Literal["normal", "italic", "oblique"]
    fontvariant: Literal["normal", "small-caps"]
    variant: Literal["normal", "small-caps"]
    fontweight: int | Literal["ultralight", "light", "normal", "regular", "book", "medium", "roman", "semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black"]
    weight: int | Literal["ultralight", "light", "normal", "regular", "book", "medium", "roman", "semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black"]
    gid: str
    horizontalalignment: Literal["left", "center", "right"]
    ha: Literal["left", "center", "right"]
    in_layout: bool
    label: object
    linespacing: float
    math_fontfamily: str
    mouseover: bool
    multialignment: Literal["left", "right", "center"]
    ma: Literal["left", "right", "center"]
    parse_math: bool
    path_effects: list[AbstractPathEffect]
    picker: bool | float | None | Callable
    position: tuple[float, float]
    rasterized: bool
    rotation: float | Literal["vertical", "horizontal"]
    rotation_mode: None | Literal["default", "anchor"]
    sketch_params: SketchParams
    snap: bool | None
    text: object
    transform: Transform
    transform_rotates_text: bool
    url: str
    usetex: bool
    verticalalignment: Literal["baseline", "bottom", "center", "center_baseline", "top"]
    va: Literal["baseline", "bottom", "center", "center_baseline", "top"]
    visible: bool
    wrap: bool
    zorder: float

class TextPropertiesWithPositions(TextPropertiesBase, total=False):
    x: float
    y: float
