# Corrected _typing.pyi stub for pandas 2.x
# Fixes issue #1657: SequenceNotStr protocol compatibility with list[str]
#
# This provides the corrected SequenceNotStr protocol definition that matches
# the fix in pandas main branch (to be released in pandas 3.0).
#
# Key fix: All parameters in index() are position-only (/ at end),
# matching the actual list.index signature.

from typing import Any, Iterator, Protocol, Sequence, TypeVar, Union, overload
from typing_extensions import SupportsIndex

import numpy as np

_T_co = TypeVar("_T_co", covariant=True)

class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    # FIXED: All parameters position-only to match list.index
    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int: ...
    def count(self, value: Any, /) -> int: ...
    def __reversed__(self) -> Iterator[_T_co]: ...

# Type aliases needed for DataFrame
Axes = Union[SequenceNotStr[Any], range, np.ndarray[Any, Any]]
Dtype = Any
