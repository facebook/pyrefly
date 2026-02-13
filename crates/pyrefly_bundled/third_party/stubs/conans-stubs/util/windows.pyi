from conans.client.tools.oss import OSInfo as OSInfo
from conans.errors import ConanException as ConanException
from conans.util.env_reader import get_env as get_env
from conans.util.files import decode_text as decode_text, load as load, mkdir as mkdir, rmdir as rmdir, save as save
from conans.util.log import logger as logger
from conans.util.sha import sha256 as sha256

CONAN_LINK: str
CONAN_REAL_PATH: str

def conan_expand_user(path): ...
def path_shortener(path, short_paths): ...
def rm_conandir(path) -> None: ...
def hashed_redirect(base, path, min_length: int = ..., attempts: int = ...): ...
