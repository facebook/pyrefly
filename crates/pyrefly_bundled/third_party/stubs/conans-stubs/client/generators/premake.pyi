from _typeshed import Incomplete
from conans.model import Generator as Generator
from conans.paths import BUILD_INFO_PREMAKE as BUILD_INFO_PREMAKE

class PremakeDeps:
    include_paths: Incomplete
    lib_paths: Incomplete
    bin_paths: Incomplete
    libs: Incomplete
    system_libs: Incomplete
    defines: Incomplete
    cxxflags: Incomplete
    cflags: Incomplete
    sharedlinkflags: Incomplete
    exelinkflags: Incomplete
    frameworks: Incomplete
    rootpath: Incomplete
    def __init__(self, deps_cpp_info) -> None: ...

class PremakeGenerator(Generator):
    @property
    def filename(self): ...
    @property
    def content(self): ...
