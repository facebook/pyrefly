from typing import Any

from .._models import Request, Response
from .base import AsyncBaseTransport

class ASGITransport(AsyncBaseTransport):
    def __init__(
        self,
        app: Any,
        raise_app_exceptions: bool = ...,
        root_path: str = ...,
        client: tuple[str, int] = ...,
    ) -> None: ...
    async def handle_async_request(self, request: Request) -> Response: ...
