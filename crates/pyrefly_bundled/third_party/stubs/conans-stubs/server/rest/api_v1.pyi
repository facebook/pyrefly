from _typeshed import Incomplete
from bottle import Bottle
from conans.errors import EXCEPTION_CODE_MAPPING as EXCEPTION_CODE_MAPPING
from conans.server.rest.bottle_plugins.http_basic_authentication import HttpBasicAuthentication as HttpBasicAuthentication
from conans.server.rest.bottle_plugins.jwt_authentication import JWTAuthentication as JWTAuthentication
from conans.server.rest.bottle_plugins.return_handler import ReturnHandlerPlugin as ReturnHandlerPlugin
from conans.server.rest.controller.common.ping import PingController as PingController
from conans.server.rest.controller.common.users import UsersController as UsersController
from conans.server.rest.controller.v1.conan import ConanController as ConanController
from conans.server.rest.controller.v1.delete import DeleteController as DeleteController
from conans.server.rest.controller.v1.file_upload_download import FileUploadDownloadController as FileUploadDownloadController
from conans.server.rest.controller.v1.search import SearchController as SearchController

class ApiV1(Bottle):
    credentials_manager: Incomplete
    updown_auth_manager: Incomplete
    server_capabilities: Incomplete
    def __init__(self, credentials_manager, updown_auth_manager, server_capabilities, *argc, **argv) -> None: ...
    def setup(self) -> None: ...
    def install_plugins(self) -> None: ...
