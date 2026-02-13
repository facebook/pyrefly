from conans.assets.templates.new_v2_cmake import source_cpp as source_cpp, source_h as source_h, test_main as test_main

conanfile_sources_v2: str
test_conanfile_v2: str
_bazel_build_test: str
_bazel_build: str
_bazel_workspace: str
_test_bazel_workspace: str
conanfile_exe: str
test_conanfile_exe_v2: str
_bazel_build_exe: str

def get_bazel_lib_files(name, version, package_name: str = ...): ...
def get_bazel_exe_files(name, version, package_name: str = ...): ...
