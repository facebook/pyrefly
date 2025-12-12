from typing import TypeAlias

from pandas import Series
from pandas.core.arrays import ExtensionArray

ABCSeries: TypeAlias = type[Series]
ABCExtensionArray: TypeAlias = type[ExtensionArray]
