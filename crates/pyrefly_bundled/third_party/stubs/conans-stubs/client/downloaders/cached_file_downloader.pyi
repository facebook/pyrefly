from _typeshed import Incomplete
from collections.abc import Generator
from conans.client.downloaders.file_downloader import check_checksum as check_checksum
from conans.errors import ConanException as ConanException
from conans.util.files import clean_dirty as clean_dirty, is_dirty as is_dirty, mkdir as mkdir, remove as remove, set_dirty as set_dirty
from conans.util.locks import SimpleLock as SimpleLock
from conans.util.log import logger as logger

class CachedFileDownloader:
    _thread_locks: Incomplete
    _cache_folder: Incomplete
    _file_downloader: Incomplete
    _user_download: Incomplete
    def __init__(self, cache_folder, file_downloader, user_download: bool = ...) -> None: ...
    def _lock(self, lock_id) -> Generator[None, None, None]: ...
    def download(self, url, file_path: Incomplete | None = ..., md5: Incomplete | None = ..., sha1: Incomplete | None = ..., sha256: Incomplete | None = ..., **kwargs): ...
    def _get_hash(self, url, checksum: Incomplete | None = ...): ...
