from typing import Any, Optional

from .._models import Request, Response
from .base import BaseTransport

class WSGITransport(BaseTransport):
    def __init__(
        self,
        app: Any,
        raise_app_exceptions: bool = ...,
        script_name: str = ...,
        remote_addr: str = ...,
        wsgi_errors: Optional[Any] = ...,
    ) -> None: ...
    def handle_request(self, request: Request) -> Response: ...
