from _typeshed import Incomplete
from conans.errors import ConanException as ConanException, ConanMigrationError as ConanMigrationError
from conans.model.version import Version as Version
from conans.util.files import load as load, save as save

CONAN_VERSION: str

class Migrator:
    conf_path: Incomplete
    current_version: Incomplete
    file_version_path: Incomplete
    out: Incomplete
    def __init__(self, conf_path, current_version, out) -> None: ...
    def migrate(self) -> None: ...
    def _make_migrations(self, old_version) -> None: ...
    def _update_version_file(self) -> None: ...
    def _load_old_version(self): ...
