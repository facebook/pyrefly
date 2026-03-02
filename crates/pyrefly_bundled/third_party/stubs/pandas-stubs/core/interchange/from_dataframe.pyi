from typing import Any

import pandas as pd

# TODO: resolve with https://github.com/pandas-dev/pandas/issues/63111
def from_dataframe(df: Any, allow_copy: bool = True) -> pd.DataFrame: ...
