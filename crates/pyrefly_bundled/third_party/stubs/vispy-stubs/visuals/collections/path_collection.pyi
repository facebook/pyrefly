# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
from .agg_fast_path_collection import AggFastPathCollection
from .agg_path_collection import AggPathCollection
from .raw_path_collection import RawPathCollection

def PathCollection(mode="agg", *args, **kwargs): ...
