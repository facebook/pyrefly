from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Int, MatrixLike
from ..base import OneToOneFeatureMixin
from ._encoders import _BaseEncoder

class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder):
    encodings_: list[ndarray]
    categories_: list[ndarray]
    target_type_: str
    target_mean_: float
    n_features_in_: int
    feature_names_in_: ndarray
    classes_: ndarray | None

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        categories: list[ArrayLike] | Literal["auto"] = "auto",
        target_type: Literal["auto", "continuous", "binary", "multiclass"] = "auto",
        smooth: Literal["auto"] | float = "auto",
        cv: int = 5,
        shuffle: bool = True,
        random_state: Int | None = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: ArrayLike) -> Self: ...
    def fit_transform(self, X: MatrixLike, y: ArrayLike) -> ndarray: ...
    def transform(self, X: MatrixLike) -> ndarray: ...
    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> ndarray: ...
    def __sklearn_tags__(self) -> dict: ...
