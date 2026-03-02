from _typeshed import Incomplete

_cmake_single_dep_vars: str

def _cmake_string_representation(value): ...
def _build_type_str(build_type): ...
def cmake_user_info_vars(deps_user_info): ...
def cmake_dependency_vars(name, deps, build_type: str = ...): ...

_cmake_package_info: str

def cmake_package_info(name, version): ...
def cmake_settings_info(settings): ...
def cmake_dependencies(dependencies, build_type: str = ...): ...

_cmake_multi_dep_vars: str

def cmake_global_vars(deps, build_type: str = ...): ...

_target_template: str

def generate_targets_section(dependencies, generator_name): ...

class CMakeCommonMacros:
    conan_message: Incomplete
    conan_get_policy: Incomplete
    conan_find_libraries_abs_path: Incomplete
    conan_package_library_targets: Incomplete
    conan_set_libcxx: Incomplete
    conan_set_std: Incomplete
    conan_set_rpath: Incomplete
    conan_set_fpic: Incomplete
    conan_output_dirs_setup: Incomplete
    conan_split_version: Incomplete
    conan_error_compiler_version: Incomplete
    conan_get_compiler: Incomplete
    check_compiler_version: Incomplete
    conan_check_compiler: Incomplete
    conan_set_flags: Incomplete
    conan_global_flags: Incomplete
    conan_target_link_libraries: Incomplete
    conan_include_build_modules: Incomplete
    conan_set_vs_runtime: Incomplete
    conan_set_vs_runtime_preserve_build_type: Incomplete
    conan_set_find_paths: Incomplete
    conan_set_find_paths_multi: Incomplete
    conan_set_find_library_paths: Incomplete
    apple_frameworks_macro: Incomplete

_cmake_common_macros: Incomplete

def _conan_basic_setup_common(addtional_macros, cmake_multi: bool = ...): ...

cmake_macros: Incomplete
cmake_macros_multi: Incomplete
