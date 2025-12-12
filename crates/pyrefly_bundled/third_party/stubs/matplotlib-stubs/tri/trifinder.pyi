import numpy as np
from numpy.typing import ArrayLike

from .triangulation import Triangulation

class TriFinder:
    def __init__(self, triangulation: Triangulation) -> None: ...

class TrapezoidMapTriFinder(TriFinder):
    def __init__(self, triangulation: Triangulation) -> None: ...
    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray: ...
