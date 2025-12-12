from typing import Literal

from matplotlib.axes import Axes
from numpy.typing import ArrayLike

def tripcolor(
    ax: Axes,
    *args,
    alpha=...,
    norm=...,
    cmap=...,
    vmin=...,
    vmax=...,
    shading: Literal["flat", "gouraud"] = "flat",
    facecolors: ArrayLike = ...,
    **kwargs,
): ...
