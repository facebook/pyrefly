from typing import Any, Literal

from sympy.plotting.plot import BaseSeries, Plot
from sympy.utilities.decorator import doctest_depends_on

class ImplicitSeries(BaseSeries):
    is_implicit = ...
    def __init__(
        self, expr, var_start_end_x, var_start_end_y, has_equality, use_interval_math, depth, nb_of_points, line_color
    ) -> None: ...
    def get_raster(
        self,
    ) -> (
        tuple[list[Any], Literal["fill"]] | tuple[Any, Any, Any, Literal["contour"]] | tuple[Any, Any, Any, Literal["contourf"]]
    ): ...

@doctest_depends_on(modules=("matplotlib",))
def plot_implicit(
    expr, x_var=..., y_var=..., adaptive=..., depth=..., points=..., line_color=..., show=..., **kwargs
) -> Plot: ...
