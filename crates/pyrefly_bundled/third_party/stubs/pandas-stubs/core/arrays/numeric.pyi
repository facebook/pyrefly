from pandas.core.arrays.masked import BaseMaskedArray

from pandas._libs.properties import cache_readonly

from pandas.core.dtypes.dtypes import BaseMaskedDtype

class NumericDtype(BaseMaskedDtype): ...

class NumericArray(BaseMaskedArray):
    @cache_readonly
    def dtype(self) -> NumericDtype: ...
