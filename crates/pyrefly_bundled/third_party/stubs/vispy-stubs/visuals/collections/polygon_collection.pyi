# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
from .raw_polygon_collection import RawPolygonCollection

# from . agg_polygon_collection import AggPolygonCollection
# from . agg_fast_polygon_collection import AggPolygonCollection

def PolygonCollection(mode="raw", *args, **kwargs): ...
