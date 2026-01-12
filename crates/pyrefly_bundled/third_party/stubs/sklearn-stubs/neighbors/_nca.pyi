from typing import Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin

class NeighborhoodComponentsAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    random_state_: RandomState = ...
    n_iter_: int = ...
    n_features_in_: int = ...
    components_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        init: Literal["auto", "pca", "lda", "identity", "random"] | MatrixLike = "auto",
        warm_start: bool = False,
        max_iter: Int = 50,
        tol: Float = 1e-5,
        callback: None | Callable = None,
        verbose: Int = 0,
        random_state: None | RandomState | int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: ArrayLike) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
