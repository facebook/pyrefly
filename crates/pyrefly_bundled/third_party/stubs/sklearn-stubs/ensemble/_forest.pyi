from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin
from ..preprocessing import OneHotEncoder
from ..tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeRegressor,
)
from ._base import BaseEnsemble

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "RandomTreesEmbedding",
]

MAX_INT = ...

class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators: int = 100,
        *,
        estimator_params=...,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs=None,
        random_state=None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight=None,
        max_samples=None,
        base_estimator: str = "deprecated",
    ) -> None: ...
    def apply(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def decision_path(self, X: MatrixLike | ArrayLike) -> tuple[spmatrix, ndarray]: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    @property
    def feature_importances_(self) -> ndarray: ...

class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators: int = 100,
        *,
        estimator_params=...,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs=None,
        random_state=None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight=None,
        max_samples=None,
        base_estimator: str = "deprecated",
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def predict_log_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators: int = 100,
        *,
        estimator_params=...,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs=None,
        random_state=None,
        verbose: int = 0,
        warm_start: bool = False,
        max_samples=None,
        base_estimator: str = "deprecated",
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class RandomForestClassifier(ForestClassifier):
    oob_decision_function_: ndarray = ...
    oob_score_: float = ...
    feature_importances_: ndarray = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_classes_: list | int = ...
    classes_: ndarray = ...
    estimators_: list[DecisionTreeClassifier] = ...
    base_estimator_: DecisionTreeClassifier = ...
    estimator_: DecisionTreeClassifier = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_estimators: Int = 100,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | Literal["sqrt", "log2"] = "sqrt",
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
        class_weight: Literal["balanced", "balanced_subsample"] | None | Mapping | Sequence[Mapping] = None,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ) -> None: ...

class RandomForestRegressor(ForestRegressor):
    oob_prediction_: ndarray = ...
    oob_score_: float = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    feature_importances_: ndarray = ...
    estimators_: list[DecisionTreeRegressor] = ...
    base_estimator_: DecisionTreeRegressor = ...
    estimator_: DecisionTreeRegressor = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_estimators: Int = 100,
        *,
        criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | Literal["sqrt", "log2"] = 1.0,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ) -> None: ...

class ExtraTreesClassifier(ForestClassifier):
    oob_decision_function_: ndarray = ...
    oob_score_: float = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    feature_importances_: ndarray = ...
    n_classes_: list | int = ...
    classes_: ndarray = ...
    estimators_: list[DecisionTreeClassifier] = ...
    base_estimator_: ExtraTreesClassifier = ...
    estimator_: ExtraTreesClassifier = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_estimators: Int = 100,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | Literal["sqrt", "log2"] = "sqrt",
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
        class_weight: Literal["balanced", "balanced_subsample"] | None | Mapping | Sequence[Mapping] = None,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ) -> None: ...

class ExtraTreesRegressor(ForestRegressor):
    oob_prediction_: ndarray = ...
    oob_score_: float = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    feature_importances_: ndarray = ...
    estimators_: list[DecisionTreeRegressor] = ...
    base_estimator_: ExtraTreeRegressor = ...
    estimator_: ExtraTreeRegressor = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_estimators: Int = 100,
        *,
        criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | Literal["sqrt", "log2"] = 1.0,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ) -> None: ...

class RandomTreesEmbedding(TransformerMixin, BaseForest):
    one_hot_encoder_: OneHotEncoder = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    feature_importances_: ndarray = ...
    estimators_: list[ExtraTreeRegressor] = ...
    base_estimator_: ExtraTreeRegressor = ...
    estimator_: ExtraTreeRegressor = ...

    _parameter_constraints: ClassVar[dict] = ...

    criterion: ClassVar[str] = ...
    max_features: ClassVar[int] = ...

    def __init__(
        self,
        n_estimators: Int = 100,
        *,
        max_depth: Int = 5,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        sparse_output: bool = True,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def fit_transform(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> spmatrix: ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> spmatrix: ...
