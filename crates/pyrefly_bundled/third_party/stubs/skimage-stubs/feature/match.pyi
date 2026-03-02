import numpy as np
from scipy.spatial.distance import cdist

def match_descriptors(
    descriptors1,
    descriptors2,
    metric=None,
    p: int = 2,
    max_distance: float = ...,
    cross_check: bool = True,
    max_ratio: float = 1.0,
): ...
