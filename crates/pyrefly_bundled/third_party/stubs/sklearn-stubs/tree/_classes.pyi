from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from ..utils._bunch import Bunch
from ._tree import (
    Tree,
)

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = ...
DOUBLE = ...

CRITERIA_CLF: dict = ...
CRITERIA_REG: dict = ...

DENSE_SPLITTERS: dict = ...

SPARSE_SPLITTERS: dict = ...

# =============================================================================
# Base decision tree
# =============================================================================

class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def get_depth(self) -> int: ...
    def get_n_leaves(self) -> int: ...
    def fit(
        self,
        X,
        y: ndarray,
        sample_weight: None | ndarray = None,
        check_input: bool = True,
    ): ...
    def predict(self, X: MatrixLike | ArrayLike, check_input: bool = True) -> ndarray: ...
    def apply(self, X: MatrixLike | ArrayLike, check_input: bool = True) -> ArrayLike: ...
    def decision_path(self, X: MatrixLike | ArrayLike, check_input: bool = True) -> spmatrix: ...
    def cost_complexity_pruning_path(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Bunch: ...
    @property
    def feature_importances_(self) -> ndarray: ...

# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    tree_: Tree = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_classes_: int | list[int] = ...
    max_features_: int = ...
    feature_importances_: ndarray = ...
    classes_: ndarray | list[ndarray] = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        splitter: Literal["best", "random"] = "best",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | None | Literal["auto", "sqrt", "log2"] = None,
        random_state: RandomState | None | Int = None,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        class_weight: None | Mapping | str | Sequence[Mapping] = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
        check_input: bool = True,
    ) -> DecisionTreeClassifier | ExtraTreeClassifier: ...
    def predict_proba(self, X: MatrixLike | ArrayLike, check_input: bool = True) -> ndarray | list[ndarray]: ...
    def predict_log_proba(self, X: MatrixLike | ArrayLike) -> ndarray | list[ndarray]: ...

class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    tree_: Tree = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    max_features_: int = ...
    feature_importances_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = "squared_error",
        splitter: Literal["best", "random"] = "best",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | None | Literal["auto", "sqrt", "log2"] = None,
        random_state: RandomState | None | Int = None,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
        check_input: bool = True,
    ) -> Self: ...

class ExtraTreeClassifier(DecisionTreeClassifier):
    tree_: Tree = ...
    n_outputs_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    feature_importances_: ndarray = ...
    n_classes_: int | list[int] = ...
    max_features_: int = ...
    classes_: ndarray | list[ndarray] = ...

    def __init__(
        self,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        splitter: Literal["random", "best"] = "random",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | None | Literal["auto", "sqrt", "log2"] = "sqrt",
        random_state: RandomState | None | Int = None,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        class_weight: None | Mapping | str | Sequence[Mapping] = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...

class ExtraTreeRegressor(DecisionTreeRegressor):
    tree_: Tree = ...
    n_outputs_: int = ...
    feature_importances_: ndarray = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    max_features_: int = ...

    def __init__(
        self,
        *,
        criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = "squared_error",
        splitter: Literal["random", "best"] = "random",
        max_depth: None | Int = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | None | Literal["auto", "sqrt", "log2"] = 1.0,
        random_state: RandomState | None | Int = None,
        min_impurity_decrease: Float = 0.0,
        max_leaf_nodes: None | Int = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
