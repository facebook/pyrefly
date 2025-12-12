from _typeshed import Incomplete
from conans.model import Generator as Generator

BUILD_INFO_QMAKE: str

class DepsCppQmake:
    include_paths: Incomplete
    lib_paths: Incomplete
    bin_paths: Incomplete
    res_paths: Incomplete
    build_paths: Incomplete
    libs: Incomplete
    system_libs: Incomplete
    frameworks: Incomplete
    framework_paths: Incomplete
    defines: Incomplete
    cxxflags: Incomplete
    cflags: Incomplete
    sharedlinkflags: Incomplete
    exelinkflags: Incomplete
    rootpath: Incomplete
    def __init__(self, cpp_info) -> None: ...

class QmakeGenerator(Generator):
    @property
    def filename(self): ...
    @property
    def content(self): ...
