# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
from .agg_segment_collection import AggSegmentCollection
from .raw_segment_collection import RawSegmentCollection

def SegmentCollection(mode="agg-fast", *args, **kwargs): ...
