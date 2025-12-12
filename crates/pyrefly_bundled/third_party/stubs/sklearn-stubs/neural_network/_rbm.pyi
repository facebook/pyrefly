from typing import ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

class BernoulliRBM(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    h_samples_: ArrayLike = ...
    components_: ArrayLike = ...
    intercept_visible_: ArrayLike = ...
    intercept_hidden_: ArrayLike = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: Int = 256,
        *,
        learning_rate: Float = 0.1,
        batch_size: Int = 10,
        n_iter: Int = 10,
        verbose: Int = 0,
        random_state: RandomState | None | Int = None,
    ) -> None: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def gibbs(self, v: ArrayLike) -> ndarray: ...
    def partial_fit(self, X: ArrayLike, y: None | MatrixLike | ArrayLike = None) -> Self: ...
    def score_samples(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: None | MatrixLike | ArrayLike = None,
    ) -> Self: ...
