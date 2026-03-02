from _typeshed import Incomplete
from conans.client.conanfile.build import run_build_method as run_build_method
from conans.client.tools import chdir as chdir
from conans.errors import ConanException as ConanException, NotFoundException as NotFoundException, conanfile_exception_formatter as conanfile_exception_formatter
from conans.model.conan_file import get_env_context_manager as get_env_context_manager
from conans.paths import CONANFILE as CONANFILE, CONANFILE_TXT as CONANFILE_TXT
from conans.util.files import mkdir as mkdir
from conans.util.log import logger as logger

def cmd_build(app, conanfile_path, base_path, source_folder, build_folder, package_folder, install_folder, test: bool = ..., should_configure: bool = ..., should_build: bool = ..., should_install: bool = ..., should_test: bool = ..., layout_source_folder: Incomplete | None = ..., layout_build_folder: Incomplete | None = ..., conf: Incomplete | None = ...) -> None: ...
