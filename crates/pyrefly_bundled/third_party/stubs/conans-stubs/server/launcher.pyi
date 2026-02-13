from _typeshed import Incomplete
from conans import REVISIONS as REVISIONS, SERVER_CAPABILITIES as SERVER_CAPABILITIES
from conans.paths import conan_expand_user as conan_expand_user
from conans.server.conf import get_server_store as get_server_store
from conans.server.crypto.jwt.jwt_credentials_manager import JWTCredentialsManager as JWTCredentialsManager
from conans.server.crypto.jwt.jwt_updown_manager import JWTUpDownAuthManager as JWTUpDownAuthManager
from conans.server.migrate import migrate_and_get_server_config as migrate_and_get_server_config
from conans.server.plugin_loader import load_authentication_plugin as load_authentication_plugin, load_authorization_plugin as load_authorization_plugin
from conans.server.rest.server import ConanServer as ConanServer
from conans.server.service.authorize import BasicAuthenticator as BasicAuthenticator, BasicAuthorizer as BasicAuthorizer

class ServerLauncher:
    force_migration: Incomplete
    server: Incomplete
    def __init__(self, force_migration: bool = ..., server_dir: Incomplete | None = ...) -> None: ...
    def launch(self) -> None: ...
