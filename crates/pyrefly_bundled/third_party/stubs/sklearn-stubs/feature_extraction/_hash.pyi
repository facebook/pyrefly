from collections.abc import Iterator
from typing import Any, ClassVar
from typing_extensions import Self

from numpy import dtype
from scipy.sparse import spmatrix

from .._typing import Int
from ..base import BaseEstimator, TransformerMixin

# Author: Lars Buitinck
# License: BSD 3 clause

class FeatureHasher(TransformerMixin, BaseEstimator):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_features: Int = ...,
        *,
        input_type: str = "dict",
        dtype: dtype = ...,
        alternate_sign: bool = True,
    ) -> None: ...
    def fit(self, X: Any = None, y: Any = None) -> Self: ...
    def transform(self, raw_X: Iterator[Any] | Iterator[Iterator]) -> spmatrix: ...
