from _typeshed import Incomplete
from conans.client.cmd.new_ci import ci_get_files as ci_get_files
from conans.errors import ConanException as ConanException
from conans.model.ref import ConanFileReference as ConanFileReference, get_reference_fields as get_reference_fields
from conans.util.files import load as load

conanfile: str
conanfile_bare: str
conanfile_sources: str
conanfile_header: str
test_conanfile: str
test_cmake: str
test_cmake_pure_c: str
test_main: str
hello_c: str
hello_h: str
hello_cpp: str
cmake_pure_c: str
cmake: str
gitignore_template: str

def _render_template(text, name, version, package_name, defines): ...
def _get_files_from_template_dir(template_dir, name, version, package_name, defines): ...
def cmd_new(ref, header: bool = ..., pure_c: bool = ..., test: bool = ..., exports_sources: bool = ..., bare: bool = ..., visual_versions: Incomplete | None = ..., linux_gcc_versions: Incomplete | None = ..., linux_clang_versions: Incomplete | None = ..., osx_clang_versions: Incomplete | None = ..., shared: Incomplete | None = ..., upload_url: Incomplete | None = ..., gitignore: Incomplete | None = ..., gitlab_gcc_versions: Incomplete | None = ..., gitlab_clang_versions: Incomplete | None = ..., circleci_gcc_versions: Incomplete | None = ..., circleci_clang_versions: Incomplete | None = ..., circleci_osx_versions: Incomplete | None = ..., template: Incomplete | None = ..., cache: Incomplete | None = ..., defines: Incomplete | None = ...): ...
