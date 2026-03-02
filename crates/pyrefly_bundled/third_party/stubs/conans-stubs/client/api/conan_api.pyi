from _typeshed import Incomplete
from conans.cli.output import ConanOutput as ConanOutput
from conans.client.cache.cache import ClientCache as ClientCache
from conans.client.migrations import ClientMigrator as ClientMigrator
from conans.client.tools.env import environment_append as environment_append
from conans.client.userio import UserIO as UserIO
from conans.errors import NoRemoteAvailable as NoRemoteAvailable
from conans.model.version import Version as Version
from conans.paths import get_conan_user_home as get_conan_user_home
from conans.util.files import exception_message_safe as exception_message_safe

def api_method(f): ...

class ConanAPIV2:
    out: Incomplete
    user_io: Incomplete
    cache_folder: Incomplete
    http_requester: Incomplete
    runner: Incomplete
    def __init__(self, cache_folder: Incomplete | None = ..., quiet: bool = ..., user_io: Incomplete | None = ..., http_requester: Incomplete | None = ..., runner: Incomplete | None = ...) -> None: ...
    def user_list(self, remote_name: Incomplete | None = ...): ...
    def user_add(self, remote_name, user_name, user_password, force_add: bool = ...): ...
    def user_remove(self, remote_name): ...
    def user_update(self, user_name, user_pasword): ...
    def search_recipes(self, query, remote_patterns: Incomplete | None = ..., local_cache: bool = ...): ...
Conan = ConanAPIV2
