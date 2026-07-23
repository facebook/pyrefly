import enum
from contextlib import contextmanager
from types import TracebackType
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

from ._auth import Auth
from ._config import Limits, Timeout
from ._models import Cookies, Headers, Request, Response
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._types import (
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxyTypes,
    QueryParamTypes,
    RequestContent,
    RequestExtensions,
    RequestFiles,
    TimeoutTypes,
    URLTypes,
    VerifyTypes,
)
from ._urls import URL, QueryParams

class UseClientDefault: ...

USE_CLIENT_DEFAULT: UseClientDefault

class ClientState(enum.Enum):
    UNOPENED = 1
    OPENED = 2
    CLOSED = 3

class BaseClient:
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        max_redirects: int = ...,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]] = ...,
        base_url: URLTypes = ...,
        trust_env: bool = ...,
        default_encoding: Union[str, Callable[[bytes], str]] = ...,
    ) -> None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def trust_env(self) -> bool: ...
    @property
    def timeout(self) -> Timeout: ...
    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None: ...
    @property
    def event_hooks(self) -> dict[str, List[Callable[..., Any]]]: ...
    @event_hooks.setter
    def event_hooks(self, event_hooks: Mapping[str, List[Callable[..., Any]]]) -> None: ...
    @property
    def auth(self) -> Optional[Auth]: ...
    @auth.setter
    def auth(self, auth: AuthTypes) -> None: ...
    @property
    def base_url(self) -> URL: ...
    @base_url.setter
    def base_url(self, url: URLTypes) -> None: ...
    @property
    def headers(self) -> Headers: ...
    @headers.setter
    def headers(self, headers: HeaderTypes) -> None: ...
    @property
    def cookies(self) -> Cookies: ...
    @cookies.setter
    def cookies(self, cookies: CookieTypes) -> None: ...
    @property
    def params(self) -> QueryParams: ...
    @params.setter
    def params(self, params: QueryParamTypes) -> None: ...
    @property
    def follow_redirects(self) -> bool: ...
    @follow_redirects.setter
    def follow_redirects(self, value: bool) -> None: ...
    def build_request(
        self,
        method: str,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Request: ...

class Client(BaseClient):
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        verify: VerifyTypes = ...,
        cert: Optional[CertTypes] = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: Optional[ProxyTypes] = ...,
        mounts: Optional[Mapping[str, Optional[BaseTransport]]] = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]] = ...,
        base_url: URLTypes = ...,
        transport: Optional[BaseTransport] = ...,
        trust_env: bool = ...,
        default_encoding: Union[str, Callable[[bytes], str]] = ...,
    ) -> None: ...
    def __enter__(self) -> "Client": ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = ...,
        exc_value: Optional[BaseException] = ...,
        traceback: Optional[TracebackType] = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def send(
        self,
        request: Request,
        *,
        stream: bool = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
    ) -> Response: ...
    def request(
        self,
        method: str,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    @contextmanager
    def stream(
        self,
        method: str,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Iterator[Response]: ...
    def get(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def options(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def head(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def post(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def put(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def patch(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def delete(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...

class AsyncClient(BaseClient):
    def __init__(
        self,
        *,
        auth: Optional[AuthTypes] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        verify: VerifyTypes = ...,
        cert: Optional[CertTypes] = ...,
        http1: bool = ...,
        http2: bool = ...,
        proxy: Optional[ProxyTypes] = ...,
        mounts: Optional[Mapping[str, Optional[AsyncBaseTransport]]] = ...,
        timeout: TimeoutTypes = ...,
        follow_redirects: bool = ...,
        limits: Limits = ...,
        max_redirects: int = ...,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]] = ...,
        base_url: URLTypes = ...,
        transport: Optional[AsyncBaseTransport] = ...,
        trust_env: bool = ...,
        default_encoding: Union[str, Callable[[bytes], str]] = ...,
    ) -> None: ...
    async def __aenter__(self) -> "AsyncClient": ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = ...,
        exc_value: Optional[BaseException] = ...,
        traceback: Optional[TracebackType] = ...,
    ) -> None: ...
    async def aclose(self) -> None: ...
    async def send(
        self,
        request: Request,
        *,
        stream: bool = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
    ) -> Response: ...
    async def request(
        self,
        method: str,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    def stream(
        self,
        method: str,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> "_AsyncStreamContextManager": ...
    async def get(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def options(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def head(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def post(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def put(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def patch(
        self,
        url: URLTypes,
        *,
        content: Optional[RequestContent] = ...,
        data: Optional[Mapping[str, Any]] = ...,
        files: Optional[RequestFiles] = ...,
        json: Optional[Any] = ...,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...
    async def delete(
        self,
        url: URLTypes,
        *,
        params: Optional[QueryParamTypes] = ...,
        headers: Optional[HeaderTypes] = ...,
        cookies: Optional[CookieTypes] = ...,
        auth: Union[AuthTypes, UseClientDefault, None] = ...,
        follow_redirects: Union[bool, UseClientDefault] = ...,
        timeout: Union[TimeoutTypes, UseClientDefault] = ...,
        extensions: Optional[RequestExtensions] = ...,
    ) -> Response: ...

class _AsyncStreamContextManager:
    async def __aenter__(self) -> Response: ...
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = ...,
        exc_value: Optional[BaseException] = ...,
        traceback: Optional[TracebackType] = ...,
    ) -> None: ...
