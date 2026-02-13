from typing import Literal

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState

from ..._typing import ArrayLike, MatrixLike
from ...base import BaseEstimator

class PredictionErrorDisplay:
    figure_: Figure = ...
    ax_: Axes = ...
    scatter_: Artist = ...
    errors_lines_: Artist | None = ...
    line_: Artist = ...

    def __init__(self, *, y_true: ArrayLike, y_pred: ArrayLike) -> None: ...
    def plot(
        self,
        ax: None | Axes = None,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"] = "residual_vs_predicted",
        scatter_kwargs: None | dict = None,
        line_kwargs: None | dict = None,
    ) -> PredictionErrorDisplay: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"] = "residual_vs_predicted",
        subsample: float | None = 1_000,
        random_state: None | RandomState | int = None,
        ax: None | Axes = None,
        scatter_kwargs: None | dict = None,
        line_kwargs: None | dict = None,
    ) -> PredictionErrorDisplay: ...
    @classmethod
    def from_predictions(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        kind: Literal["actual_vs_predicted", "residual_vs_predicted"] = "residual_vs_predicted",
        subsample: float | None = 1_000,
        random_state: None | RandomState | int = None,
        ax: None | Axes = None,
        scatter_kwargs: None | dict = None,
        line_kwargs: None | dict = None,
    ) -> PredictionErrorDisplay: ...
