from _typeshed import Incomplete
from collections.abc import Generator
from conans.client.loader import parse_conanfile as parse_conanfile
from conans.client.recorder.action_recorder import ActionRecorder as ActionRecorder
from conans.errors import ConanException as ConanException, NotFoundException as NotFoundException
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.model.requires import Requirement as Requirement
from conans.util.conan_v2_mode import conan_v2_error as conan_v2_error
from typing import NamedTuple

class PythonRequire(NamedTuple):
    ref: Incomplete
    module: Incomplete
    conanfile: Incomplete
    exports_folder: Incomplete
    exports_sources_folder: Incomplete

class PyRequire:
    module: Incomplete
    conanfile: Incomplete
    ref: Incomplete
    path: Incomplete
    def __init__(self, module, conanfile, ref, path) -> None: ...

class PyRequires:
    _pyrequires: Incomplete
    _transitive: Incomplete
    def __init__(self) -> None: ...
    def update_transitive(self, conanfile) -> None: ...
    def all_items(self): ...
    def all_refs(self): ...
    def items(self): ...
    def __getitem__(self, item): ...
    def __setitem__(self, key, value) -> None: ...

class PyRequireLoader:
    _proxy: Incomplete
    _range_resolver: Incomplete
    _cached_py_requires: Incomplete
    def __init__(self, proxy, range_resolver) -> None: ...
    _check_updates: Incomplete
    _update: Incomplete
    _remotes: Incomplete
    def enable_remotes(self, check_updates: bool = ..., update: bool = ..., remotes: Incomplete | None = ...) -> None: ...
    def capture_requires(self) -> Generator[Incomplete, None, None]: ...
    def load_py_requires(self, conanfile, lock_python_requires, loader) -> None: ...
    def _resolve_py_requires(self, py_requires_refs, lock_python_requires, loader): ...
    def _resolve_ref(self, py_requires_ref, lock_python_requires): ...
    def _load_pyreq_conanfile(self, loader, lock_python_requires, ref): ...

class ConanPythonRequire:
    _generator_manager: Incomplete
    _cached_requires: Incomplete
    _proxy: Incomplete
    _range_resolver: Incomplete
    _requires: Incomplete
    valid: bool
    _check_updates: bool
    _update: bool
    _remote_name: Incomplete
    locked_versions: Incomplete
    def __init__(self, proxy, range_resolver, generator_manager: Incomplete | None = ...) -> None: ...
    _remotes: Incomplete
    def enable_remotes(self, check_updates: bool = ..., update: bool = ..., remotes: Incomplete | None = ...) -> None: ...
    def capture_requires(self) -> Generator[Incomplete, None, None]: ...
    def _look_for_require(self, reference): ...
    def __call__(self, reference): ...
