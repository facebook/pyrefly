import pathlib
from collections.abc import Hashable, Sequence
from typing import Any, Callable, Literal, TypeAlias, TypeVar

from . import path
from ._enums import CapStyle, JoinStyle
from .artist import Artist
from .backend_bases import RendererBase
from .markers import MarkerStyle
from .transforms import Bbox, Transform

RGBColorType: TypeAlias = tuple[float, float, float] | str
RGBAColorType: TypeAlias = (
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    |
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float]
    |
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float]
)

ColorType: TypeAlias = RGBColorType | RGBAColorType

RGBColourType: TypeAlias = RGBColorType
RGBAColourType: TypeAlias = RGBAColorType
ColourType: TypeAlias = ColorType

LineStyleType: TypeAlias = str | tuple[float, Sequence[float]]
DrawStyleType: TypeAlias = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
MarkEveryType: TypeAlias = None | int | tuple[int, int] | slice | list[int] | float | tuple[float, float] | list[bool]

MarkerType: TypeAlias = str | path.Path | MarkerStyle
FillStyleType: TypeAlias = Literal["full", "left", "right", "bottom", "top", "none"]
JoinStyleType: TypeAlias = JoinStyle | Literal["miter", "round", "bevel"]
CapStyleType: TypeAlias = CapStyle | Literal["butt", "projecting", "round"]

CoordsBaseType: TypeAlias = str | Artist | Transform | Callable[[RendererBase], Bbox | Transform]
CoordsType: TypeAlias = CoordsBaseType | tuple[CoordsBaseType, CoordsBaseType]

RcStyleType: TypeAlias = str | dict[str, Any] | pathlib.Path | Sequence[str | pathlib.Path | dict[str, Any]]

_HT = TypeVar("_HT", bound=Hashable)
HashableList: TypeAlias = list[_HT | HashableList[_HT]]
"""A nested list of Hashable values."""
