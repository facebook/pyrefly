import numpy as np
from numpy.typing import ArrayLike

from .triangulation import Triangulation
from .triinterpolate import TriInterpolator

class TriRefiner:
    def __init__(self, triangulation: Triangulation) -> None: ...

class UniformTriRefiner(TriRefiner):
    def __init__(self, triangulation: Triangulation) -> None: ...
    def refine_triangulation(
        self,
        return_tri_index: bool = False,
        subdiv: int = 3,
    ) -> tuple[Triangulation, np.ndarray]: ...
    def refine_field(
        self,
        z: ArrayLike,
        triinterpolator: TriInterpolator = ...,
        subdiv: int = 3,
    ) -> tuple[Triangulation, np.ndarray]: ...
