from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    TypeAlias,
)

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.table import Table
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._typing import (
    HashableT,
    npt,
)

_Color: TypeAlias = str | Sequence[float]

def table(
    ax: Axes,
    data: DataFrame | Series,
    **kwargs: Any,
) -> Table: ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: Literal["hist", "kde"] = "hist",
    marker: str = ".",
    density_kwds: dict[str, Any] | None = None,
    hist_kwds: dict[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwargs: Any,
) -> npt.NDArray[np.object_]: ...
def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = None,
    color: _Color | Sequence[_Color] | None = None,
    colormap: str | Colormap | None = None,
    **kwds: Any,
) -> Axes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = None,
    samples: int = 200,
    color: _Color | Sequence[_Color] | None = None,
    colormap: str | Colormap | None = None,
    **kwargs: Any,
) -> Axes: ...
def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds: Any,
) -> Figure: ...
def parallel_coordinates(
    frame: DataFrame,
    class_column: Hashable,
    cols: list[HashableT] | None = None,
    ax: Axes | None = None,
    color: _Color | Sequence[_Color] | None = None,
    use_columns: bool = False,
    xticks: Sequence[float] | None = None,
    colormap: str | Colormap | None = None,
    axvlines: bool = True,
    axvlines_kwds: dict[str, Any] | None = None,
    sort_labels: bool = False,
    **kwargs: Any,
) -> Axes: ...
def lag_plot(
    series: Series, lag: int = 1, ax: Axes | None = None, **kwds: Any
) -> Axes: ...
def autocorrelation_plot(
    series: Series, ax: Axes | None = None, **kwargs: Any
) -> Axes: ...

plot_params: dict[str, Any]
