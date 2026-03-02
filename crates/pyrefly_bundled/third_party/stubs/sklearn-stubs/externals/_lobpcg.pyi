from typing import Callable

from numpy import ndarray
from scipy.sparse.linalg import LinearOperator

from .._typing import ArrayLike, Float, Int, MatrixLike, Scalar

__all__ = ["lobpcg"]

def lobpcg(
    A: MatrixLike | LinearOperator | ArrayLike | Callable,
    X: ArrayLike | Float,
    B: ArrayLike | None | Callable | MatrixLike | LinearOperator = None,
    M: ArrayLike | None | Callable | MatrixLike | LinearOperator = None,
    Y: float | None | ArrayLike = None,
    tol: None | Scalar = None,
    maxiter: None | int = None,
    largest: None | bool = True,
    verbosityLevel: Int = 0,
    retLambdaHistory: None | bool = False,
    retResidualNormsHistory: None | bool = False,
    restartControl: None | int = 20,
) -> tuple[ndarray, ndarray, ndarray, ndarray]: ...
