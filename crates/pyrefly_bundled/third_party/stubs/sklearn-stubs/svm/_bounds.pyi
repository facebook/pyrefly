from typing import Literal

from .._typing import ArrayLike, Float, MatrixLike

def l1_min_c(
    X: MatrixLike | ArrayLike,
    y: ArrayLike,
    *,
    loss: Literal["squared_hinge", "log"] = "squared_hinge",
    fit_intercept: bool = True,
    intercept_scaling: Float = 1.0,
) -> Float: ...
