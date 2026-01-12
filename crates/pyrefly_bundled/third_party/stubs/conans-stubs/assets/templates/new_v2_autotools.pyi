from _typeshed import Incomplete
from conans.assets.templates.new_v2_cmake import source_cpp as source_cpp, source_h as source_h, test_main as test_main

conanfile_lib: Incomplete
configure_ac: Incomplete
makefile_am: Incomplete
makefile_am_lib: Incomplete
test_conanfile: Incomplete
test_configure_ac: Incomplete
test_makefile_am: Incomplete

def get_autotools_lib_files(name, version, package_name: str = ...): ...

conanfile_exe: Incomplete
test_conanfile_exe: Incomplete
makefile_am_exe: Incomplete

def get_autotools_exe_files(name, version, package_name: str = ...): ...
