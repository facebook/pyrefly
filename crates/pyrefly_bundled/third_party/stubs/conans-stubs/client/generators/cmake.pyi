from _typeshed import Incomplete
from conans.client.generators.cmake_common import CMakeCommonMacros as CMakeCommonMacros, cmake_dependencies as cmake_dependencies, cmake_dependency_vars as cmake_dependency_vars, cmake_global_vars as cmake_global_vars, cmake_macros as cmake_macros, cmake_package_info as cmake_package_info, cmake_settings_info as cmake_settings_info, cmake_user_info_vars as cmake_user_info_vars, generate_targets_section as generate_targets_section
from conans.model import Generator as Generator
from conans.paths import BUILD_INFO_CMAKE as BUILD_INFO_CMAKE

class DepsCppCmake:
    include_paths: Incomplete
    include_path: Incomplete
    lib_paths: Incomplete
    res_paths: Incomplete
    bin_paths: Incomplete
    build_paths: Incomplete
    src_paths: Incomplete
    framework_paths: Incomplete
    libs: Incomplete
    system_libs: Incomplete
    frameworks: Incomplete
    defines: Incomplete
    compile_definitions: Incomplete
    cxxflags: Incomplete
    cflags: Incomplete
    sharedlinkflags: Incomplete
    exelinkflags: Incomplete
    cxxflags_list: Incomplete
    cflags_list: Incomplete
    sharedlinkflags_list: Incomplete
    exelinkflags_list: Incomplete
    rootpath: Incomplete
    build_modules_paths: Incomplete
    def __init__(self, cpp_info, generator_name) -> None: ...

class CMakeGenerator(Generator):
    name: str
    @property
    def filename(self): ...
    @property
    def content(self): ...
