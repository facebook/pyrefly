from _typeshed import Incomplete
from conans.client.rest import response_to_str as response_to_str
from conans.client.tools.files import check_md5 as check_md5, check_sha1 as check_sha1, check_sha256 as check_sha256
from conans.errors import AuthenticationException as AuthenticationException, ConanConnectionError as ConanConnectionError, ConanException as ConanException, ForbiddenException as ForbiddenException, NotFoundException as NotFoundException, RequestErrorException as RequestErrorException
from conans.util import progress_bar as progress_bar
from conans.util.files import mkdir as mkdir
from conans.util.log import logger as logger
from conans.util.tracer import log_download as log_download

def check_checksum(file_path, md5, sha1, sha256) -> None: ...

class FileDownloader:
    _output: Incomplete
    _requester: Incomplete
    _verify_ssl: Incomplete
    _config_retry: Incomplete
    _config_retry_wait: Incomplete
    def __init__(self, requester, output, verify, config_retry, config_retry_wait) -> None: ...
    def download(self, url, file_path: Incomplete | None = ..., auth: Incomplete | None = ..., retry: Incomplete | None = ..., retry_wait: Incomplete | None = ..., overwrite: bool = ..., headers: Incomplete | None = ..., md5: Incomplete | None = ..., sha1: Incomplete | None = ..., sha256: Incomplete | None = ...): ...
    def _download_file(self, url, auth, headers, file_path, try_resume: bool = ...): ...

def _call_with_retry(out, retry, retry_wait, method, *args, **kwargs): ...
