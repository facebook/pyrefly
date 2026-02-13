from numpy import ndarray
from numpy.random import RandomState

from .._typing import Int
from ..utils._bunch import Bunch

# The original data can be found at:
# https://cs.nyu.edu/~roweis/data/olivettifaces.mat
FACES = ...

def fetch_olivetti_faces(
    *,
    data_home: None | str = None,
    shuffle: bool = False,
    random_state: RandomState | None | Int = 0,
    download_if_missing: bool = True,
    return_X_y: bool = False,
) -> Bunch | tuple[Bunch, tuple] | tuple[ndarray, ndarray]: ...
