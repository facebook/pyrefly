# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
import re
import weakref
from typing import TypeVar

from .. import visuals
from ..visuals.filters import Alpha, PickingFilter
from .node import Node

_T = TypeVar("_T")

class VisualNode(Node):
    _next_id: int = ...
    _visual_ids = ...

    def __init__(self, parent: Node | None = None, name: str | None = None): ...
    def _update_opacity(self): ...
    def _set_clipper(self, node, clipper): ...
    @property
    def picking(self): ...
    @picking.setter
    def picking(self, p): ...
    def _update_trsys(self, event): ...
    @property
    def interactive(self): ...
    @interactive.setter
    def interactive(self, i): ...
    def draw(self): ...

def create_visual_node(subclass: _T) -> _T: ...
def generate_docstring(subclass, clsname): ...

# This is _not_ automated to help with auto-completion of IDEs,
# python REPL and IPython.
# Explicitly initializing these members allow IDEs to lookup
# and provide auto-completion. One problem is the fact that
# Docstrings are _not_ looked up correctly by IDEs, since they
# are attached programatically in the create_visual_node call.
# However, help(vispy.scene.FooVisual) still works

Arrow = visuals.ArrowVisual
Axis = visuals.AxisVisual
Box = visuals.BoxVisual
ColorBar = visuals.ColorBarVisual
Compound = visuals.CompoundVisual
Cube = visuals.CubeVisual
Ellipse = visuals.EllipseVisual
Graph = visuals.GraphVisual
GridLines = visuals.GridLinesVisual
GridMesh = visuals.GridMeshVisual
Histogram = visuals.HistogramVisual
Image = visuals.ImageVisual
ComplexImage = visuals.ComplexImageVisual
InfiniteLine = visuals.InfiniteLineVisual
Isocurve = visuals.IsocurveVisual
Isoline = visuals.IsolineVisual
Isosurface = visuals.IsosurfaceVisual
Line = visuals.LineVisual
LinearRegion = visuals.LinearRegionVisual
LinePlot = visuals.LinePlotVisual
Markers = visuals.MarkersVisual
Mesh = visuals.MeshVisual
MeshNormals = visuals.MeshNormalsVisual
Plane = visuals.PlaneVisual
Polygon = visuals.PolygonVisual
Rectangle = visuals.RectangleVisual
RegularPolygon = visuals.RegularPolygonVisual
ScrollingLines = visuals.ScrollingLinesVisual
Spectrogram = visuals.SpectrogramVisual
Sphere = visuals.SphereVisual
SurfacePlot = visuals.SurfacePlotVisual
Text = visuals.TextVisual
Tube = visuals.TubeVisual
# Visual = create_visual_node(visuals.Visual)  # Should not be created
Volume = visuals.VolumeVisual
Windbarb = visuals.WindbarbVisual
XYZAxis = visuals.XYZAxisVisual
