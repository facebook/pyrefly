from conans.model.ref import ConanFileReference as ConanFileReference
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.service.common.search import SearchService as SearchService

class SearchController:
    @staticmethod
    def attach_to(app): ...
