from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Int, MatrixLike
from ._base import MissingIndicator, _BaseImputer

# Authors: Ashim Bhattarai <ashimb9@gmail.com>
#          Thomas J Fan <thomasjpfan@gmail.com>
# License: BSD 3 clause

class KNNImputer(_BaseImputer):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    indicator_: MissingIndicator = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        missing_values: float | None | str = ...,
        n_neighbors: Int = 5,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        metric: Callable | Literal["nan_euclidean"] = "nan_euclidean",
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
    ) -> None: ...
    def fit(self, X: list[list[int | float]] | MatrixLike, y: Any = None) -> Self: ...
    def transform(self, X: list[list[int | float]] | MatrixLike) -> ndarray: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
