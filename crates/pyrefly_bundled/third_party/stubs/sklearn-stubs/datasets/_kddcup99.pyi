from typing import Literal

from numpy.random import RandomState

from .._typing import Int
from ..utils._bunch import Bunch

# The original data can be found at:
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz
ARCHIVE = ...

# The original data can be found at:
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz
ARCHIVE_10_PERCENT = ...

logger = ...

def fetch_kddcup99(
    *,
    subset: None | Literal["SA", "SF", "http", "smtp"] = None,
    data_home: None | str = None,
    shuffle: bool = False,
    random_state: RandomState | None | Int = None,
    percent10: bool = True,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | tuple[Bunch, tuple]: ...
