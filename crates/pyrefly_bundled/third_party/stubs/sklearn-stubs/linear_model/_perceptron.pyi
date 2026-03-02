from collections.abc import Mapping
from typing import ClassVar, Literal

from numpy import ndarray
from numpy.random import RandomState

from .._typing import Float, Int
from ._sgd_fast import LossFunction
from ._stochastic_gradient import BaseSGDClassifier

# Author: Mathieu Blondel
# License: BSD 3 clause

class Perceptron(BaseSGDClassifier):
    t_: int = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    loss_function_: LossFunction = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        penalty: None | Literal["l2", "l1", "elasticnet"] = None,
        alpha: Float = 0.0001,
        l1_ratio: Float = 0.15,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        shuffle: bool = True,
        verbose: Int = 0,
        eta0: Float = 1.0,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = 0,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        n_iter_no_change: Int = 5,
        class_weight: Mapping[str, float] | None | str = None,
        warm_start: bool = False,
    ) -> None: ...
