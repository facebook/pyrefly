# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

from .agg_point_collection import AggPointCollection
from .raw_point_collection import RawPointCollection

def PointCollection(mode="raw", *args, **kwargs): ...
