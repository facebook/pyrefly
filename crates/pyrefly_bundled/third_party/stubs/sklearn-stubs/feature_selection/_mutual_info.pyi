from typing import Literal

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Int, MatrixLike

# Author: Nikolay Mayorov <n59_ru@hotmail.com>
# License: 3-clause BSD

def mutual_info_regression(
    X: MatrixLike,
    y: ArrayLike,
    *,
    discrete_features: Literal["auto"] | ArrayLike | bool = "auto",
    n_neighbors: Int = 3,
    copy: bool = True,
    random_state: RandomState | None | Int = None,
) -> ndarray: ...
def mutual_info_classif(
    X: MatrixLike,
    y: ArrayLike,
    *,
    discrete_features: Literal["auto"] | ArrayLike | bool = "auto",
    n_neighbors: Int = 3,
    copy: bool = True,
    random_state: RandomState | None | Int = None,
) -> ndarray: ...
