from conans.model.version import Version as Version
from conans.server.conf import ConanServerConfigParser as ConanServerConfigParser
from conans.server.migrations import ServerMigrator as ServerMigrator
from conans.util.log import logger as logger

def migrate_and_get_server_config(base_folder, force_migration: bool = ..., is_custom_path: bool = ...): ...
