from collections.abc import Mapping, Sequence
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import RandomState

from ..._typing import ArrayLike, Int, MatrixLike
from ...base import BaseEstimator
from ...utils._bunch import Bunch

class PartialDependenceDisplay:
    figure_: Figure = ...
    heatmaps_: ndarray = ...
    bars_: ndarray = ...
    contours_: ndarray = ...
    deciles_hlines_: ndarray = ...
    deciles_vlines_: ndarray = ...
    lines_: ndarray = ...
    axes_: ndarray = ...
    bounding_ax_: None | Axes = ...

    def __init__(
        self,
        pd_results: list[Bunch] | Sequence[Bunch],
        *,
        features: Sequence[int] | Sequence[tuple[int, int]],
        feature_names: Sequence[str],
        target_idx: Int,
        deciles: dict,
        pdp_lim: None | Mapping | str = "deprecated",
        kind: (Sequence[Literal["average", "individual", "both"]] | Literal["average", "individual", "both"]) = "average",
        subsample: float | None = 1000,
        random_state: RandomState | None | Int = None,
        is_categorical: None | ArrayLike = None,
    ) -> None: ...
    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: MatrixLike,
        features: list[int | str | tuple[str, str]] | Sequence[int | str | Sequence[int | str]],
        *,
        categorical_features: None | MatrixLike | ArrayLike = None,
        feature_names: None | ArrayLike = None,
        target: None | Int = None,
        response_method: Literal["auto", "predict_proba", "decision_function"] = "auto",
        n_cols: Int = 3,
        grid_resolution: Int = 100,
        percentiles: tuple[float, ...] = ...,
        method: str = "auto",
        n_jobs: None | Int = None,
        verbose: Int = 0,
        line_kw: None | dict = None,
        ice_lines_kw: None | dict = None,
        pd_line_kw: None | dict = None,
        contour_kw: None | dict = None,
        ax: None | Sequence[Axes] | Axes = None,
        kind: Literal["average", "individual", "both"] = "average",
        centered: bool = False,
        subsample: float | None = 1000,
        random_state: RandomState | None | Int = None,
    ) -> PartialDependenceDisplay: ...
    def plot(
        self,
        *,
        ax: None | Sequence[Axes] | Axes = None,
        n_cols: Int = 3,
        line_kw: None | dict = None,
        ice_lines_kw: None | dict = None,
        pd_line_kw: None | dict = None,
        contour_kw: None | dict = None,
        bar_kw: None | dict = None,
        heatmap_kw: None | dict = None,
        pdp_lim: None | dict = None,
        centered: bool = False,
    ) -> PartialDependenceDisplay: ...
