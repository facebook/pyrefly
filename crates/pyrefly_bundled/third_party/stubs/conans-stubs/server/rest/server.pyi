from _typeshed import Incomplete
from conans.server.rest.api_v1 import ApiV1 as ApiV1
from conans.server.rest.api_v2 import ApiV2 as ApiV2

class ConanServer:
    store: Incomplete
    root_app: Incomplete
    run_port: Incomplete
    api_v1: Incomplete
    api_v2: Incomplete
    def __init__(self, run_port, credentials_manager, updown_auth_manager, authorizer, authenticator, server_store, server_capabilities) -> None: ...
    def run(self, **kwargs) -> None: ...
