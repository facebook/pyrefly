from typing import Callable, Literal

from numpy import ndarray

from ..._typing import Float

__all__ = ["consensus_score"]

def consensus_score(
    a: tuple[ndarray, ndarray] | tuple[int, int],
    b: tuple[ndarray, ndarray] | tuple[int, int],
    *,
    similarity: Literal["jaccard"] | Callable = "jaccard",
) -> Float: ...
