from _typeshed import Incomplete
from conans.client.build.compiler_flags import format_frameworks as format_frameworks
from conans.model import Generator as Generator
from conans.paths import BUILD_INFO_XCODE as BUILD_INFO_XCODE

class XCodeGenerator(Generator):
    template: str
    include_dirs: Incomplete
    lib_dirs: Incomplete
    libs: Incomplete
    definitions: Incomplete
    c_compiler_flags: Incomplete
    cxx_compiler_flags: Incomplete
    linker_flags: Incomplete
    rootpaths: Incomplete
    frameworks: Incomplete
    framework_paths: Incomplete
    system_libs: Incomplete
    def __init__(self, conanfile) -> None: ...
    @property
    def filename(self): ...
    @property
    def content(self): ...
