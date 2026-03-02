from conans import DEFAULT_REVISION_V1 as DEFAULT_REVISION_V1
from conans.errors import NotFoundException as NotFoundException, PackageNotFoundException as PackageNotFoundException, RecipeNotFoundException as RecipeNotFoundException
from conans.model.ref import ConanFileReference as ConanFileReference, PackageReference as PackageReference
from conans.paths import CONAN_MANIFEST as CONAN_MANIFEST
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.service.v1.service import ConanService as ConanService

class ConanController:
    @staticmethod
    def attach_to(app): ...
