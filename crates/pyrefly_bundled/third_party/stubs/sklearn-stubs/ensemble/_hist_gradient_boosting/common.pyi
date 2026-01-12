import enum
from typing import Final

import numpy.dtypes

class MonotonicConstraint(enum.IntFlag):
    NEG = -1
    NO_CST = 0
    POS = 1

ALMOST_INF: float
HISTOGRAM_DTYPE: numpy.dtypes.VoidDType
NEG: Final = MonotonicConstraint.NEG
NO_CST: Final = MonotonicConstraint.NO_CST
POS: Final = MonotonicConstraint.POS
PREDICTOR_RECORD_DTYPE: numpy.dtypes.VoidDType
__test__: dict
