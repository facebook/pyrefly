from _typeshed import Incomplete
from conans import DEFAULT_REVISION_V1 as DEFAULT_REVISION_V1
from conans.migrations import Migrator as Migrator
from conans.model.version import Version as Version
from conans.paths import PACKAGES_FOLDER as PACKAGES_FOLDER
from conans.server.revision_list import RevisionList as RevisionList
from conans.server.store.server_store import REVISIONS_FILE as REVISIONS_FILE
from conans.util.files import list_folder_subdirs as list_folder_subdirs, mkdir as mkdir, rmdir as rmdir, save as save
from conans.util.log import logger as logger

class ServerMigrator(Migrator):
    force_migrations: Incomplete
    store_path: Incomplete
    def __init__(self, conf_path, store_path, current_version, out, force_migrations) -> None: ...
    def _make_migrations(self, old_version) -> None: ...
    def migrate_to_revisions_layout(self) -> None: ...
