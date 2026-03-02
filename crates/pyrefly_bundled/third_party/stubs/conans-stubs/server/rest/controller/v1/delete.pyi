from conans import DEFAULT_REVISION_V1 as DEFAULT_REVISION_V1
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.service.v1.service import ConanService as ConanService

class DeleteController:
    @staticmethod
    def attach_to(app) -> None: ...
