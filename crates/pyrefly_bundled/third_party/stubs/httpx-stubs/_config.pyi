import ssl
from typing import Any, Optional

from ._types import CertTypes, URLTypes, VerifyTypes

DEFAULT_TIMEOUT_CONFIG: "Timeout"
DEFAULT_LIMITS: "Limits"
DEFAULT_MAX_REDIRECTS: int

def create_ssl_context(
    cert: Optional[CertTypes] = ...,
    verify: VerifyTypes = ...,
    trust_env: bool = ...,
    http2: bool = ...,
) -> ssl.SSLContext: ...

class Timeout:
    connect: Optional[float]
    read: Optional[float]
    write: Optional[float]
    pool: Optional[float]
    def __init__(
        self,
        timeout: Any = ...,
        *,
        connect: Optional[float] = ...,
        read: Optional[float] = ...,
        write: Optional[float] = ...,
        pool: Optional[float] = ...,
    ) -> None: ...
    def as_dict(self) -> dict[str, Optional[float]]: ...
    def __eq__(self, other: Any) -> bool: ...

class Limits:
    max_connections: Optional[int]
    max_keepalive_connections: Optional[int]
    keepalive_expiry: Optional[float]
    def __init__(
        self,
        *,
        max_connections: Optional[int] = ...,
        max_keepalive_connections: Optional[int] = ...,
        keepalive_expiry: Optional[float] = ...,
    ) -> None: ...
    def __eq__(self, other: Any) -> bool: ...

class Proxy:
    url: Any
    headers: Any
    ssl_context: Optional[ssl.SSLContext]
    auth: Optional[tuple[str, str]]
    def __init__(
        self,
        url: URLTypes,
        *,
        ssl_context: Optional[ssl.SSLContext] = ...,
        auth: Optional[tuple[str, str]] = ...,
        headers: Any = ...,
    ) -> None: ...
