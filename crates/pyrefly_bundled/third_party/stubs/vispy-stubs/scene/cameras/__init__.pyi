# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
__all__ = [
    "ArcballCamera",
    "BaseCamera",
    "FlyCamera",
    "MagnifyCamera",
    "Magnify1DCamera",
    "PanZoomCamera",
    "TurntableCamera",
]

from ._base import make_camera as make_camera
from .arcball import ArcballCamera as ArcballCamera
from .base_camera import BaseCamera as BaseCamera
from .fly import FlyCamera as FlyCamera
from .magnify import Magnify1DCamera as Magnify1DCamera, MagnifyCamera as MagnifyCamera
from .panzoom import PanZoomCamera as PanZoomCamera
from .turntable import TurntableCamera as TurntableCamera
