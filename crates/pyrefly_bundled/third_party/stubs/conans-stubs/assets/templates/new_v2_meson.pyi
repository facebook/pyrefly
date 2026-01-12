from conans.assets.templates.new_v2_cmake import source_cpp as source_cpp, source_h as source_h, test_main as test_main

conanfile_sources_v2: str
test_conanfile_v2: str
_meson_build_test: str
_meson_build: str

def get_meson_lib_files(name, version, package_name: str = ...): ...

conanfile_exe: str
test_conanfile_exe_v2: str
_meson_build_exe: str

def get_meson_exe_files(name, version, package_name: str = ...): ...
