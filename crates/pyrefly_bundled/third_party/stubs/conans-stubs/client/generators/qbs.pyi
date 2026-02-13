from _typeshed import Incomplete
from conans.model import Generator as Generator
from conans.paths import BUILD_INFO_QBS as BUILD_INFO_QBS

class DepsCppQbs:
    include_paths: Incomplete
    lib_paths: Incomplete
    libs: Incomplete
    framework_paths: Incomplete
    frameworks: Incomplete
    defines: Incomplete
    cxxflags: Incomplete
    cflags: Incomplete
    linkerFlags: Incomplete
    bin_paths: Incomplete
    rootpath: Incomplete
    def __init__(self, cpp_info) -> None: ...

class QbsGenerator(Generator):
    name: str
    @property
    def filename(self): ...
    @property
    def content(self): ...
