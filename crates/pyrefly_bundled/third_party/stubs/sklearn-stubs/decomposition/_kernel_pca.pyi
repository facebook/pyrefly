from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Sylvain Marie <sylvain.marie@schneider-electric.com>
# License: BSD 3 clause

class KernelPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    X_fit_: ndarray = ...
    X_transformed_fit_: ndarray = ...
    dual_coef_: ndarray = ...
    eigenvectors_: ndarray = ...
    eigenvalues_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        kernel: Callable | Literal["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"] = "linear",
        gamma: None | Float = None,
        degree: Int = 3,
        coef0: Float = 1,
        kernel_params: None | dict = None,
        alpha: Float = 1.0,
        fit_inverse_transform: bool = False,
        eigen_solver: Literal["auto", "dense", "arpack", "randomized"] = "auto",
        tol: Float = 0,
        max_iter: None | Int = None,
        iterated_power: Literal["auto"] | int = "auto",
        remove_zero_eig: bool = False,
        random_state: RandomState | None | Int = None,
        copy_X: bool = True,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike | ArrayLike, y: Any = None) -> Self: ...
    def fit_transform(self, X: MatrixLike | ArrayLike, y: Any = None, **params) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def inverse_transform(self, X: MatrixLike) -> ndarray: ...
