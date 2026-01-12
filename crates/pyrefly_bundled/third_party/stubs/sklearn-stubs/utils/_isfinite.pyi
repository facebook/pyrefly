from enum import IntEnum
from typing import Final

import numpy as np

__test__: dict

class FiniteStatus(IntEnum):
    all_finite = 0
    has_nan = 1
    has_infinite = 2

all_finite: Final = FiniteStatus.all_finite
has_nan: Final = FiniteStatus.has_nan
has_infinite: Final = FiniteStatus.has_infinite

def cy_isfinite(a: np.ndarray, allow_nan: bool = False) -> bool: ...
