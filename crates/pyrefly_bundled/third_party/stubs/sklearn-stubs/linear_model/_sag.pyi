from typing import Literal

from numpy import ndarray
from numpy.random.mtrand import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike

# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#
# License: BSD 3 clause

def get_auto_step_size(
    max_squared_sum: Float,
    alpha_scaled: Float,
    loss: Literal["log", "squared", "multinomial"],
    fit_intercept: bool,
    n_samples: None | Int = None,
    is_saga: bool = False,
) -> Float: ...
def sag_solver(
    X: MatrixLike | ArrayLike,
    y: ArrayLike,
    sample_weight: None | ArrayLike = None,
    loss: Literal["log", "squared", "multinomial"] = "log",
    alpha: Float = 1.0,
    beta: Float = 0.0,
    max_iter: Int = 1000,
    tol: Float = 0.001,
    verbose: Int = 0,
    random_state: None | Int | RandomState = None,
    check_input: bool = True,
    max_squared_sum: None | Float = None,
    warm_start_mem: dict[str, ndarray] | None | dict = None,
    is_saga: bool = False,
) -> tuple[ndarray, int, dict] | tuple[ndarray, int, dict[str, ndarray | int]]: ...
