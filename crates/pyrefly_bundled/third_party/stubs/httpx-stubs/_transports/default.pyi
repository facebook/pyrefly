from typing import Any, Optional

from .._config import Limits
from .._models import Request, Response
from .._types import CertTypes, ProxyTypes, VerifyTypes
from .base import AsyncBaseTransport, BaseTransport

class HTTPTransport(BaseTransport):
    def __init__(
        self,
        verify: VerifyTypes = ...,
        cert: Optional[CertTypes] = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
        trust_env: bool = ...,
        proxy: Optional[ProxyTypes] = ...,
        uds: Optional[str] = ...,
        local_address: Optional[str] = ...,
        retries: int = ...,
        socket_options: Optional[Any] = ...,
    ) -> None: ...
    def handle_request(self, request: Request) -> Response: ...

class AsyncHTTPTransport(AsyncBaseTransport):
    def __init__(
        self,
        verify: VerifyTypes = ...,
        cert: Optional[CertTypes] = ...,
        http1: bool = ...,
        http2: bool = ...,
        limits: Limits = ...,
        trust_env: bool = ...,
        proxy: Optional[ProxyTypes] = ...,
        uds: Optional[str] = ...,
        local_address: Optional[str] = ...,
        retries: int = ...,
        socket_options: Optional[Any] = ...,
    ) -> None: ...
    async def handle_async_request(self, request: Request) -> Response: ...
