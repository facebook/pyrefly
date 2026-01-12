from _typeshed import Incomplete
from conans.client.generators.cmake_common import CMakeCommonMacros as CMakeCommonMacros

target_template: str

def find_transitive_dependencies(public_deps_filenames, find_modules): ...

class CMakeFindPackageCommonMacros:
    conan_message: Incomplete
    apple_frameworks_macro: Incomplete
    conan_package_library_targets: Incomplete
