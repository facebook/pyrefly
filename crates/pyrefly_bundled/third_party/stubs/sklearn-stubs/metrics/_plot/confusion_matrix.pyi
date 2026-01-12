from typing import Literal

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy import ndarray

from ..._typing import ArrayLike, MatrixLike
from ...base import BaseEstimator

class ConfusionMatrixDisplay:
    figure_: Figure = ...
    ax_: Axes = ...
    text_: None | ndarray = ...
    im_: AxesImage = ...

    def __init__(self, confusion_matrix: MatrixLike, *, display_labels: None | ArrayLike = None) -> None: ...
    def plot(
        self,
        *,
        include_values: bool = True,
        cmap: Colormap | str = "viridis",
        xticks_rotation: float | Literal["vertical", "horizontal"] = "horizontal",
        values_format: None | str = None,
        ax: None | Axes = None,
        colorbar: bool = True,
        im_kw: None | dict = None,
        text_kw: None | dict = None,
    ) -> ConfusionMatrixDisplay: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        *,
        labels: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
        normalize: Literal["true", "pred", "all"] | None = None,
        display_labels: None | ArrayLike = None,
        include_values: bool = True,
        xticks_rotation: float | Literal["vertical", "horizontal"] = "horizontal",
        values_format: None | str = None,
        cmap: Colormap | str = "viridis",
        ax: None | Axes = None,
        colorbar: bool = True,
        im_kw: None | dict = None,
        text_kw: None | dict = None,
    ) -> ConfusionMatrixDisplay: ...
    @classmethod
    def from_predictions(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        labels: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
        normalize: Literal["true", "pred", "all"] | None = None,
        display_labels: None | ArrayLike = None,
        include_values: bool = True,
        xticks_rotation: float | Literal["vertical", "horizontal"] = "horizontal",
        values_format: None | str = None,
        cmap: Colormap | str = "viridis",
        ax: None | Axes = None,
        colorbar: bool = True,
        im_kw: None | dict = None,
        text_kw: None | dict = None,
    ) -> ConfusionMatrixDisplay: ...
