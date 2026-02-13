from collections.abc import Sequence
from typing import Literal

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator
from ..utils._bunch import Bunch

__all__ = [
    "partial_dependence",
]

def partial_dependence(
    estimator: BaseEstimator,
    X: MatrixLike,
    features: tuple[int] | Sequence[tuple[int, str]] | tuple[str, str] | tuple[int, int],
    *,
    categorical_features: None | MatrixLike | ArrayLike = None,
    feature_names: None | ArrayLike = None,
    response_method: Literal["auto", "predict_proba", "decision_function"] = "auto",
    percentiles: tuple[float, ...] = ...,
    grid_resolution: Int = 100,
    method: Literal["auto", "recursion", "brute"] = "auto",
    kind: Literal["average", "individual", "both"] = "average",
) -> Bunch: ...
