from typing import Literal, overload

from ..utils import Bunch

__all__ = ["fetch_openml"]

_OPENML_PREFIX: str = ...
_SEARCH_NAME: str = ...
_DATA_INFO: str = ...
_DATA_FEATURES: str = ...
_DATA_QUALITIES: str = ...
_DATA_FILE: str = ...

OpenmlQualitiesType = ...
OpenmlFeaturesType = ...

class OpenMLError(ValueError): ...

@overload
def fetch_openml(
    name: str | None = None,
    *,
    version: str | int = "active",
    data_id: int | None = None,
    data_home: str | None = None,
    target_column: str | list | None = "default-target",
    cache: bool = True,
    return_X_y: Literal[False] = ...,
    as_frame: str | bool = "auto",
    n_retries: int = 3,
    delay: float = 1.0,
    parser: str | None = "warn",
) -> Bunch: ...
@overload
def fetch_openml(
    name: str | None = None,
    *,
    version: str | int = "active",
    data_id: int | None = None,
    data_home: str | None = None,
    target_column: str | list | None = "default-target",
    cache: bool = True,
    return_X_y: Literal[True] = ...,
    as_frame: str | bool = "auto",
    n_retries: int = 3,
    delay: float = 1.0,
    parser: str | None = "warn",
) -> tuple[Bunch, tuple]: ...
