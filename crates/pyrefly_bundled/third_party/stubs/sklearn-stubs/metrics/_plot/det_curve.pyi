from typing import Literal

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..._typing import ArrayLike, MatrixLike
from ...base import BaseEstimator

class DetCurveDisplay:
    figure_: Figure = ...
    ax_: Axes = ...
    line_: Artist = ...

    def __init__(
        self,
        *,
        fpr: ArrayLike,
        fnr: ArrayLike,
        estimator_name: None | str = None,
        pos_label: None | str | int = None,
    ) -> None: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: None | ArrayLike = None,
        response_method: Literal["predict_proba", "decision_function", "auto"] = "auto",
        pos_label: None | str | int = None,
        name: None | str = None,
        ax: None | Axes = None,
        **kwargs,
    ) -> DetCurveDisplay: ...
    @classmethod
    def from_predictions(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        sample_weight: None | ArrayLike = None,
        pos_label: None | str | int = None,
        name: None | str = None,
        ax: None | Axes = None,
        **kwargs,
    ) -> DetCurveDisplay: ...
    def plot(self, ax: None | Axes = None, *, name: None | str = None, **kwargs) -> DetCurveDisplay: ...
