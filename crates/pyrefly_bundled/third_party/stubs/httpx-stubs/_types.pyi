import ssl
from typing import (
    IO,
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ._auth import Auth
from ._config import Proxy, Timeout
from ._models import Cookies, Headers, Request, Response
from ._urls import URL, QueryParams

PrimitiveData = Optional[Union[str, int, float, bool]]

URLTypes = Union[URL, str]

QueryParamTypes = Union[
    QueryParams,
    Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]],
    List[Tuple[str, PrimitiveData]],
    Tuple[Tuple[str, PrimitiveData], ...],
    str,
    bytes,
]

HeaderTypes = Union[
    Headers,
    Mapping[str, str],
    Mapping[bytes, bytes],
    Sequence[Tuple[str, str]],
    Sequence[Tuple[bytes, bytes]],
]

CookieTypes = Union[Cookies, Mapping[str, str], List[Tuple[str, str]]]

CertTypes = Union[str, Tuple[str, str], Tuple[str, str, str]]
VerifyTypes = Union[str, bool, ssl.SSLContext]
TimeoutTypes = Union[
    Optional[float],
    Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    Timeout,
]
ProxyTypes = Union[URLTypes, Proxy]

AuthTypes = Union[
    Tuple[Union[str, bytes], Union[str, bytes]],
    Callable[[Request], Request],
    Auth,
]

RequestContent = Union[str, bytes, Iterable[bytes], AsyncIterable[bytes]]
ResponseContent = Union[str, bytes, Iterable[bytes], AsyncIterable[bytes]]

RequestData = Mapping[str, Any]

FileContent = Union[IO[bytes], bytes, str]
FileTypes = Union[
    FileContent,
    Tuple[Optional[str], FileContent],
    Tuple[Optional[str], FileContent, Optional[str]],
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]
RequestFiles = Union[Mapping[str, FileTypes], Sequence[Tuple[str, FileTypes]]]

RequestExtensions = Mapping[str, Any]
ResponseExtensions = Mapping[str, Any]

class SyncByteStream:
    def __iter__(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...

class AsyncByteStream:
    def __aiter__(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...
