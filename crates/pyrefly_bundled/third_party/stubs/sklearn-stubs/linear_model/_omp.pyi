from collections.abc import Iterable
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import MultiOutputMixin, RegressorMixin
from ..model_selection import BaseCrossValidator
from ._base import LinearModel

# Author: Vlad Niculae
#
# License: BSD 3 clause

premature: str = ...

def orthogonal_mp(
    X: ArrayLike,
    y: MatrixLike | ArrayLike,
    *,
    n_nonzero_coefs: None | Int = None,
    tol: None | Float = None,
    precompute: str | bool = False,
    copy_X: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> ndarray | tuple[ndarray, int] | tuple[ndarray, ndarray | int]: ...
def orthogonal_mp_gram(
    Gram: MatrixLike,
    Xy: MatrixLike | ArrayLike,
    *,
    n_nonzero_coefs: None | Int = None,
    tol: None | Float = None,
    norms_squared: None | ArrayLike = None,
    copy_Gram: bool = True,
    copy_Xy: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> ndarray | tuple[ndarray, ndarray | int]: ...

class OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_nonzero_coefs_: int = ...
    n_iter_: ArrayLike | int = ...
    intercept_: float | ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        n_nonzero_coefs: None | Int = None,
        tol: None | Float = None,
        fit_intercept: bool = True,
        normalize: str | bool = "deprecated",
        precompute: Literal["auto"] | bool = "auto",
    ) -> None: ...
    def fit(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> Self: ...

class OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: ArrayLike | int = ...
    n_nonzero_coefs_: int = ...
    coef_: ndarray = ...
    intercept_: float | ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        copy: bool = True,
        fit_intercept: bool = True,
        normalize: str | bool = "deprecated",
        max_iter: None | Int = None,
        cv: int | BaseCrossValidator | Iterable | None = None,
        n_jobs: None | Int = None,
        verbose: int | bool = False,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: ArrayLike) -> Self: ...
