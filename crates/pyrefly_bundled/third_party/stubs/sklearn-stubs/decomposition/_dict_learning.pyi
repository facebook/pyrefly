from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin

def sparse_encode(
    X: ArrayLike,
    dictionary: MatrixLike,
    *,
    gram: None | MatrixLike = None,
    cov: None | MatrixLike = None,
    algorithm: Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"] = "lasso_lars",
    n_nonzero_coefs: None | Int = None,
    alpha: None | Float = None,
    copy_cov: bool = True,
    init: None | MatrixLike = None,
    max_iter: Int = 1000,
    n_jobs: None | Int = None,
    check_input: bool = True,
    verbose: Int = 0,
    positive: bool = False,
) -> ndarray: ...
def dict_learning(
    X: ArrayLike,
    n_components: Int,
    *,
    alpha: Int,
    max_iter: Int = 100,
    tol: Float = 1e-8,
    method: Literal["lars", "cd"] = "lars",
    n_jobs: None | Int = None,
    dict_init: None | MatrixLike = None,
    code_init: None | MatrixLike = None,
    callback: None | Callable = None,
    verbose: bool = False,
    random_state: RandomState | None | Int = None,
    return_n_iter: bool = False,
    positive_dict: bool = False,
    positive_code: bool = False,
    method_max_iter: Int = 1000,
) -> tuple[ndarray, ndarray, ndarray, int]: ...
def dict_learning_online(
    X: ArrayLike,
    n_components: None | int = 2,
    *,
    alpha: Float = 1,
    n_iter: str | Int = "deprecated",
    max_iter: None | Int = None,
    return_code: bool = True,
    dict_init: None | MatrixLike = None,
    callback: None | Callable = None,
    batch_size: str | Int = "warn",
    verbose: bool = False,
    shuffle: bool = True,
    n_jobs: None | Int = None,
    method: Literal["lars", "cd"] = "lars",
    iter_offset: str | Int = "deprecated",
    random_state: RandomState | None | Int = None,
    return_inner_stats: str | bool = "deprecated",
    inner_stats: str | tuple[ArrayLike, ArrayLike] = "deprecated",
    return_n_iter: str | bool = "deprecated",
    positive_dict: bool = False,
    positive_code: bool = False,
    method_max_iter: Int = 1000,
    tol: Float = 1e-3,
    max_no_improvement: Int = 10,
) -> tuple[ndarray, ndarray, int]: ...

class _BaseSparseCoding(ClassNamePrefixFeaturesOutMixin, TransformerMixin):
    def __init__(
        self,
        transform_algorithm: str,
        transform_n_nonzero_coefs: None | int,
        transform_alpha: float | None,
        split_sign: bool,
        n_jobs,
        positive_code: bool,
        transform_max_iter: int,
    ) -> None: ...
    def transform(self, X: ArrayLike) -> ndarray: ...

class SparseCoder(_BaseSparseCoding, BaseEstimator):
    feature_names_in_: ndarray = ...

    _required_parameters: ClassVar[list] = ...

    def __init__(
        self,
        dictionary: MatrixLike,
        *,
        transform_algorithm: Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"] = "omp",
        transform_n_nonzero_coefs: None | Int = None,
        transform_alpha: None | Float = None,
        split_sign: bool = False,
        n_jobs: None | Int = None,
        positive_code: bool = False,
        transform_max_iter: Int = 1000,
    ) -> None: ...
    def fit(self, X: Any, y: Any = None) -> Self: ...
    def transform(self, X: ArrayLike, y: Any = None) -> ndarray: ...
    @property
    def n_components_(self) -> int: ...
    @property
    def n_features_in_(self) -> int: ...

class DictionaryLearning(_BaseSparseCoding, BaseEstimator):
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    error_: ndarray = ...
    components_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        alpha: Float = 1,
        max_iter: Int = 1000,
        tol: Float = 1e-8,
        fit_algorithm: Literal["lars", "cd"] = "lars",
        transform_algorithm: Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"] = "omp",
        transform_n_nonzero_coefs: None | Int = None,
        transform_alpha: None | Float = None,
        n_jobs: None | int = None,
        code_init: None | MatrixLike = None,
        dict_init: None | MatrixLike = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: RandomState | None | Int = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: Int = 1000,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...

class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator):
    n_steps_: int = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    components_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: None | Int = None,
        *,
        alpha: Float = 1,
        n_iter: str | Int = "deprecated",
        max_iter: None | Int = None,
        fit_algorithm: Literal["lars", "cd"] = "lars",
        n_jobs: None | Int = None,
        batch_size: str | Int = "warn",
        shuffle: bool = True,
        dict_init: None | MatrixLike = None,
        transform_algorithm: Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"] = "omp",
        transform_n_nonzero_coefs: None | Int = None,
        transform_alpha: None | Float = None,
        verbose: int | bool = False,
        split_sign: bool = False,
        random_state: RandomState | None | Int = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: Int = 1000,
        callback: None | Callable = None,
        tol: Float = 1e-3,
        max_no_improvement: Int = 10,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def partial_fit(
        self,
        X: MatrixLike,
        y: Any = None,
        iter_offset: str | Int = "deprecated",
    ) -> Self: ...
