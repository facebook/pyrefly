from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator
from ._binary_tree import BinaryTree

VALID_KERNELS: list = ...

TREE_DICT: dict = ...

class KernelDensity(BaseEstimator):
    bandwidth_: float = ...
    feature_names_in_: ndarray = ...
    tree_: BinaryTree = ...
    n_features_in_: int = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        algorithm: Literal["kd_tree", "ball_tree", "auto"] = "auto",
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        metric: str = "euclidean",
        atol: Float = 0,
        rtol: Float = 0,
        breadth_first: bool = True,
        leaf_size: Int = 40,
        metric_params: None | dict = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y=None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def score_samples(self, X: MatrixLike) -> ndarray: ...
    def score(self, X: MatrixLike, y=None) -> Float: ...
    def sample(self, n_samples: Int = 1, random_state: RandomState | None | Int = None) -> ndarray: ...
