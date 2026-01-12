import numpy as np
from scipy import sparse as sparse

from .._typing import Int, MatrixLike
from ..metrics.pairwise import pairwise_distances as pairwise_distances

# Authors: Aric Hagberg <hagberg@lanl.gov>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD 3 clause

###############################################################################
# Path and connected component analysis.
# Code adapted from networkx
def single_source_shortest_path_length(graph: MatrixLike, source: Int, *, cutoff: None | Int = None) -> dict: ...
