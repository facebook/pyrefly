from ast import Add
from typing import Any, Literal

from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.geometry.line import Segment, Segment2D, Segment3D
from sympy.geometry.point import Point, Point2D, Point3D
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle
from sympy.series.order import Order

def find(x, equation): ...
def are_coplanar(*e) -> bool: ...
def are_similar(e1, e2) -> Any | Literal[True]: ...
def centroid(*args) -> None: ...
def closest_points(*args) -> set[tuple[Point2D, ...]]: ...
def convex_hull(
    *args, polygon=...
) -> (
    tuple[Any, None]
    | Point
    | Point2D
    | Point3D
    | Segment2D
    | Segment3D
    | Segment
    | tuple[Point | Point2D | Point3D | Segment2D | Segment3D | Segment, None]
    | RegularPolygon
    | Polygon
    | Triangle
    | tuple[list[Any], list[Any]]
    | None
): ...
def farthest_points(*args) -> set[tuple[Any, Any]]: ...
def idiff(eq, y, x, n=...) -> Basic | Add | Order | Mul | Any | None: ...
def intersection(*entities, pairwise=..., **kwargs) -> list[Any]: ...
