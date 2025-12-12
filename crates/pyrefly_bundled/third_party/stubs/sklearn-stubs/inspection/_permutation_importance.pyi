from collections.abc import Mapping
from typing import Any, Callable

from numpy.random import RandomState

from .._typing import ArrayLike, Int, MatrixLike
from ..utils._bunch import Bunch

def permutation_importance(
    estimator: Any,
    X: MatrixLike,
    y: None | MatrixLike | ArrayLike,
    *,
    scoring: ArrayLike | None | tuple | Callable | Mapping | str = None,
    n_repeats: Int = 5,
    n_jobs: None | int = None,
    random_state: RandomState | None | Int = None,
    sample_weight: None | ArrayLike = None,
    max_samples: float = 1.0,
) -> Bunch | dict[str, Bunch]: ...
