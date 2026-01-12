from numpy import ndarray
from numpy.random import RandomState

from .._typing import Int
from ..utils._bunch import Bunch

# The original data can be found in:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
ARCHIVE = ...

logger = ...

# Column names reference:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
FEATURE_NAMES: list = ...
TARGET_NAMES: list = ...

def fetch_covtype(
    *,
    data_home: None | str = None,
    download_if_missing: bool = True,
    random_state: RandomState | None | Int = None,
    shuffle: bool = False,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | tuple[Bunch, tuple] | tuple[ndarray, ndarray]: ...
