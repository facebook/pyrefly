from typing import Literal

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..._typing import ArrayLike, Float, MatrixLike
from ...base import BaseEstimator

class RocCurveDisplay:
    figure_: Figure = ...
    ax_: Axes = ...
    line_: Artist = ...

    def __init__(
        self,
        *,
        fpr: ArrayLike,
        tpr: ArrayLike,
        roc_auc: None | Float = None,
        estimator_name: None | str = None,
        pos_label: None | str | int = None,
    ) -> None: ...
    def plot(self, ax: None | Axes = None, *, name: None | str = None, **kwargs) -> RocCurveDisplay: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: None | ArrayLike = None,
        drop_intermediate: bool = True,
        response_method: Literal["predict_proba", "decision_function", "auto"] = "auto",
        pos_label: None | str | int = None,
        name: None | str = None,
        ax: None | Axes = None,
        **kwargs,
    ) -> RocCurveDisplay: ...
    @classmethod
    def from_predictions(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        sample_weight: None | ArrayLike = None,
        drop_intermediate: bool = True,
        pos_label: None | str | int = None,
        name: None | str = None,
        ax: None | Axes = None,
        **kwargs,
    ) -> RocCurveDisplay: ...
