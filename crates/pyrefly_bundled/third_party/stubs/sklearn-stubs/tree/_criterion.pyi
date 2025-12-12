import numpy as np

class Criterion:
    y: np.ndarray
    sample_weight: np.ndarray

    sample_indices: np.ndarray
    start: int
    pos: int
    end: int

    n_outputs: int
    n_samples: int
    n_node_samples: int
    weighted_n_samples: float
    weighted_n_node_samples: float
    weighted_n_left: float
    weighted_n_right: float

    def init(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray,
        weighted_n_samples: float,
        sample_indices: np.ndarray,
        start: int,
        end: int,
    ) -> int: ...
    def reset(self) -> int: ...
    def reverse_reset(self) -> int: ...
    def update(self, new_pos: int) -> int: ...
    def node_impurity(self) -> float: ...
    def children_impurity(self, impurity_left: list[float], impurity_right: list[float]) -> None: ...
    def node_value(self, dest: list[float]) -> None: ...
    def impurity_improvement(self, impurity_parent: float, impurity_left: float, impurity_right: float) -> float: ...
    def proxy_impurity_improvement(self) -> float: ...

class ClassificationCriterion(Criterion):
    n_classes: np.ndarray
    max_n_classes: int

    sum_total: np.ndarray
    sum_left: np.ndarray
    sum_right: np.ndarray

class RegressionCriterion(Criterion):
    sq_sum_total: float

    sum_total: np.ndarray
    sum_left: np.ndarray
    sum_right: np.ndarray
