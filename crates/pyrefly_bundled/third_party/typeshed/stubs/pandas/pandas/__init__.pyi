# Stub for pandas with corrected SequenceNotStr protocol
# This provides the fix for issue #1657 until pandas 3.0 is released
#
# NOTE: This is a minimal stub that only includes commonly used pandas exports.
# For full pandas functionality, users should wait for pandas 3.0 or use
# a complete pandas-stubs package.

from typing import Any

from pandas.core.frame import DataFrame as DataFrame
from pandas.core.series import Series as Series

# Minimal stubs for common functions - accepting Any to avoid breaking existing code
def read_csv(filepath_or_buffer: Any, **kwargs: Any) -> DataFrame: ...
def read_excel(io: Any, **kwargs: Any) -> DataFrame: ...
def read_json(path_or_buf: Any, **kwargs: Any) -> DataFrame: ...
def concat(objs: Any, **kwargs: Any) -> Any: ...

__all__ = ["DataFrame", "Series", "read_csv", "read_excel", "read_json", "concat"]
