from _typeshed import Incomplete
from conans.client import tools as tools
from conans.client.build import join_arguments as join_arguments
from conans.client.build.compiler_flags import architecture_flag as architecture_flag, build_type_define as build_type_define, build_type_flags as build_type_flags, format_defines as format_defines, format_framework_paths as format_framework_paths, format_frameworks as format_frameworks, format_include_paths as format_include_paths, format_libraries as format_libraries, format_library_paths as format_library_paths, libcxx_define as libcxx_define, libcxx_flag as libcxx_flag, pic_flag as pic_flag, rpath_flags as rpath_flags, sysroot_flag as sysroot_flag
from conans.client.build.cppstd_flags import cppstd_from_settings as cppstd_from_settings
from conans.client.tools.env import environment_append as environment_append
from conans.client.tools.oss import OSInfo as OSInfo, args_to_string as args_to_string, cpu_count as cpu_count, cross_building as cross_building, detected_architecture as detected_architecture, detected_os as detected_os, get_build_os_arch as get_build_os_arch, get_gnu_triplet as get_gnu_triplet, get_target_os_arch as get_target_os_arch
from conans.client.tools.win import unix_path as unix_path
from conans.errors import ConanException as ConanException
from conans.model.build_info import DEFAULT_BIN as DEFAULT_BIN, DEFAULT_INCLUDE as DEFAULT_INCLUDE, DEFAULT_LIB as DEFAULT_LIB, DEFAULT_SHARE as DEFAULT_SHARE
from conans.util.conan_v2_mode import conan_v2_error as conan_v2_error
from conans.util.files import get_abs_path as get_abs_path

class AutoToolsBuildEnvironment:
    _conanfile: Incomplete
    _win_bash: Incomplete
    _include_rpath_flags: Incomplete
    subsystem: Incomplete
    _deps_cpp_info: Incomplete
    _os: Incomplete
    _os_version: Incomplete
    _os_sdk: Incomplete
    _os_subsystem: Incomplete
    _arch: Incomplete
    _build_type: Incomplete
    _compiler: Incomplete
    _compiler_version: Incomplete
    _compiler_runtime: Incomplete
    _libcxx: Incomplete
    _cppstd: Incomplete
    libs: Incomplete
    include_paths: Incomplete
    library_paths: Incomplete
    defines: Incomplete
    flags: Incomplete
    cxx_flags: Incomplete
    cppstd_flag: Incomplete
    link_flags: Incomplete
    fpic: Incomplete
    def __init__(self, conanfile, win_bash: bool = ..., include_rpath_flags: bool = ...) -> None: ...
    def _configure_fpic(self): ...
    def _get_host_build_target_flags(self): ...
    def configure(self, configure_dir: Incomplete | None = ..., args: Incomplete | None = ..., build: Incomplete | None = ..., host: Incomplete | None = ..., target: Incomplete | None = ..., pkg_config_paths: Incomplete | None = ..., vars: Incomplete | None = ..., use_default_install_dirs: bool = ...) -> None: ...
    def _configure_help_output(self, configure_path): ...
    def _adjust_path(self, path): ...
    @staticmethod
    def _valid_configure_flag(varname, args, available_flags): ...
    @staticmethod
    def _is_flag_in_args(varname, args): ...
    def make(self, args: str = ..., make_program: Incomplete | None = ..., target: Incomplete | None = ..., vars: Incomplete | None = ...) -> None: ...
    def install(self, args: str = ..., make_program: Incomplete | None = ..., vars: Incomplete | None = ...) -> None: ...
    def _configure_link_flags(self): ...
    def _configure_flags(self): ...
    def _configure_cxx_flags(self): ...
    def _configure_defines(self): ...
    def _get_vars(self): ...
    @property
    def vars_dict(self): ...
    @property
    def vars(self): ...

def _environ_value_prefix(var_name, prefix: str = ...): ...
