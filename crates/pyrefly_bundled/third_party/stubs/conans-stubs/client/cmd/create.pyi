from _typeshed import Incomplete
from conans.client.cmd.test import install_build_and_test as install_build_and_test
from conans.client.manager import deps_install as deps_install
from conans.errors import ConanException as ConanException
from conans.model.ref import ConanFileReference as ConanFileReference

def _get_test_conanfile_path(tf, conanfile_path): ...
def create(app, ref, graph_info, remotes, update, build_modes, manifest_folder, manifest_verify, manifest_interactive, keep_build, test_build_folder, test_folder, conanfile_path, recorder, is_build_require: bool = ..., require_overrides: Incomplete | None = ...) -> None: ...
