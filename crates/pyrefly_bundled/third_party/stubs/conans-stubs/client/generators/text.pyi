from _typeshed import Incomplete
from conans.errors import ConanException as ConanException
from conans.model import Generator as Generator
from conans.model.build_info import CppInfo as CppInfo, DepCppInfo as DepCppInfo, DepsCppInfo as DepsCppInfo
from conans.model.env_info import DepsEnvInfo as DepsEnvInfo
from conans.model.user_info import DepsUserInfo as DepsUserInfo
from conans.paths import BUILD_INFO as BUILD_INFO
from conans.util.log import logger as logger

class RootCppTXT:
    include_paths: Incomplete
    lib_paths: Incomplete
    res_paths: Incomplete
    build_paths: Incomplete
    libs: Incomplete
    system_libs: Incomplete
    defines: Incomplete
    cxxflags: Incomplete
    cflags: Incomplete
    sharedlinkflags: Incomplete
    exelinkflags: Incomplete
    bin_paths: Incomplete
    sysroot: Incomplete
    frameworks: Incomplete
    framework_paths: Incomplete
    def __init__(self, cpp_info) -> None: ...

class DepCppTXT(RootCppTXT):
    version: Incomplete
    name: Incomplete
    rootpath: Incomplete
    generatornames: Incomplete
    generatorfilenames: Incomplete
    def __init__(self, cpp_info) -> None: ...

class TXTGenerator(Generator):
    name: str
    _USER_INFO_HOST_PREFIX: str
    _USER_INFO_BUILD_PREFIX: str
    @property
    def filename(self): ...
    @staticmethod
    def loads(text, filter_empty: bool = ...): ...
    @staticmethod
    def _loads_user_info(text, user_info_prefix): ...
    @staticmethod
    def _loads_cpp_info(text, filter_empty): ...
    @property
    def content(self): ...
