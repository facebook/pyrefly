from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from pandas.core.series import Series
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, TransformerMixin

# Author: Henry Lin <hlin117@gmail.com>
#         Tom Dupré la Tour

# License: BSD

class KBinsDiscretizer(TransformerMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_bins_: ndarray = ...
    bin_edges_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_bins: ArrayLike | Int = 5,
        *,
        encode: Literal["onehot", "onehot-dense", "ordinal"] = "onehot",
        strategy: Literal["uniform", "quantile", "kmeans"] = "quantile",
        dtype: None | Float = None,
        subsample: int | None | Literal["warn"] = "warn",
        random_state: RandomState | None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Series | None | ndarray = None) -> Self: ...
    def transform(self, X: MatrixLike) -> ndarray | spmatrix: ...
    def inverse_transform(self, Xt: MatrixLike) -> ndarray: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
