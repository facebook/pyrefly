from pandas.core.indexes.base import _IndexSubclassBase

from pandas._typing import (
    S1,
    GenericT_co,
)

class ExtensionIndex(_IndexSubclassBase[S1, GenericT_co]): ...
