from _typeshed import Incomplete
from conans.server.rest.api_v1 import ApiV1 as ApiV1
from conans.server.rest.controller.common.ping import PingController as PingController
from conans.server.rest.controller.common.users import UsersController as UsersController
from conans.server.rest.controller.v2.conan import ConanControllerV2 as ConanControllerV2
from conans.server.rest.controller.v2.delete import DeleteControllerV2 as DeleteControllerV2
from conans.server.rest.controller.v2.revisions import RevisionsController as RevisionsController
from conans.server.rest.controller.v2.search import SearchControllerV2 as SearchControllerV2

class ApiV2(ApiV1):
    credentials_manager: Incomplete
    server_capabilities: Incomplete
    def __init__(self, credentials_manager, server_capabilities) -> None: ...
    def setup(self) -> None: ...
