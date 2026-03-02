from conans.model.ref import ConanFileReference as ConanFileReference
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.rest.controller.v2 import get_package_ref as get_package_ref
from conans.server.service.v1.service import ConanService as ConanService

class DeleteControllerV2:
    @staticmethod
    def attach_to(app) -> None: ...
